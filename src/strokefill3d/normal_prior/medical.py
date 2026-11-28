"""Physical-space MRI preprocessing through the pip-installed SimpleITK API."""

import nibabel as nib
import numpy as np


def sitk_module():
    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise RuntimeError(
            "Preprocessing requires the Python extra: pip install -e '.[preprocess]'"
        ) from error
    return sitk


def to_sitk(image, data=None):
    """NiBabel RAS/xyz to ITK LPS/zyx, retaining physical voxel centers."""
    sitk = sitk_module()
    affine = image.affine
    spacing = nib.affines.voxel_sizes(affine)
    direction = affine[:3, :3] / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4):
        raise ValueError(
            "Sheared NIfTI geometry must be resampled before preprocessing"
        )
    ras_to_lps = np.diag([-1.0, -1.0, 1.0])
    array = np.asarray(image.dataobj) if data is None else np.asarray(data)
    result = sitk.GetImageFromArray(np.ascontiguousarray(array.transpose(2, 1, 0)))
    result.SetSpacing(tuple(map(float, spacing)))
    result.SetDirection(tuple((ras_to_lps @ direction).ravel()))
    result.SetOrigin(tuple(ras_to_lps @ affine[:3, 3]))
    return result


def array_from_sitk(image):
    return sitk_module().GetArrayFromImage(image).transpose(2, 1, 0)


def n4_correct(image, observed, threads=4):
    sitk = sitk_module()
    source = to_sitk(image, np.asarray(image.dataobj, dtype=np.float32))
    mask = to_sitk(image, observed.astype(np.uint8))
    small = sitk.Shrink(source, [3, 3, 3])
    small_mask = sitk.Shrink(mask, [3, 3, 3])
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetNumberOfThreads(threads)
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    corrector.SetConvergenceThreshold(1e-6)
    extent = np.array(source.GetSize()) * np.array(source.GetSpacing())
    control_points = np.maximum(np.ceil(extent / 200).astype(int), 1) + 3
    corrector.SetNumberOfControlPoints(list(map(int, control_points)))
    corrector.Execute(small, small_mask)
    log_bias = corrector.GetLogBiasFieldAsImage(source)
    bias = array_from_sitk(sitk.Exp(log_bias)).astype(np.float32)
    corrected = np.asarray(image.dataobj, dtype=np.float32) / bias
    if not np.isfinite(corrected).all() or not np.isfinite(bias).all():
        raise RuntimeError("Nonfinite N4 bias correction")
    return (
        corrected,
        bias,
        {
            "iterations": [50, 50, 30, 20],
            "shrink": 3,
            "control_points": list(map(int, control_points)),
            "convergence": 1e-6,
        },
    )


def _registration(sitk, fixed_mask, moving_mask, threads, stage):
    method = sitk.ImageRegistrationMethod()
    method.SetNumberOfThreads(threads)
    if stage == "affine":
        method.SetMetricAsCorrelation()
    else:
        method.SetMetricAsMattesMutualInformation(32)
    method.SetMetricFixedMask(fixed_mask)
    method.SetMetricMovingMask(moving_mask)
    method.SetMetricSamplingStrategy(method.REGULAR)
    method.SetMetricSamplingPercentage(0.15, seed=2026)
    method.SetInterpolator(sitk.sitkLinear)
    large = min(fixed_mask.GetSize()) >= 128
    method.SetShrinkFactorsPerLevel([8, 4, 2, 1] if large else [2, 1])
    method.SetSmoothingSigmasPerLevel([3, 2, 1, 0] if large else [1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0 if stage == "similarity" else 1.0,
        minStep=0.001,
        numberOfIterations=200,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-6,
        estimateLearningRate=method.EachIteration,
        maximumStepSizeInPhysicalUnits=2.0 if stage == "similarity" else 1.0,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.AddCommand(
        sitk.sitkMultiResolutionIterationEvent,
        lambda: print(
            f"Registration {stage}: level {method.GetCurrentLevel()}", flush=True
        ),
    )
    return method


def register_affine(fixed, moving, fixed_brain, moving_brain, observed, threads=4):
    """Return the fixed/template-to-moving/native pull transform in LPS mm."""
    sitk = sitk_module()
    fixed_mask = to_sitk(fixed, fixed_brain.astype(np.uint8))
    moving_mask = to_sitk(moving, observed.astype(np.uint8))

    def window(image, mask):
        data = np.asarray(image.dataobj, dtype=np.float32)
        low, high = np.percentile(data[mask], [0.5, 99.5])
        if high <= low:
            raise ValueError("Registration requires nonconstant observed brain tissue")
        data = np.where(mask, np.clip((data - low) / (high - low), 0, 1), 0)
        return to_sitk(image, data.astype(np.float32))

    fixed_image = window(fixed, fixed_brain)
    # Masked injury intensities must not affect windowing, initialization or MI.
    moving_image = window(moving, observed)
    initial = sitk.CenteredTransformInitializer(
        sitk.Cast(fixed_mask, sitk.sitkFloat32),
        to_sitk(moving, moving_brain.astype(np.float32)),
        sitk.Similarity3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )
    # Large scanner/template size differences can trap MI in an underscaled
    # local optimum. Start from brain physical-volume scale, then optimize the
    # seven similarity parameters and finally the twelve affine parameters.
    fixed_volume = fixed_brain.sum() * abs(np.linalg.det(fixed.affine[:3, :3]))
    moving_volume = moving_brain.sum() * abs(np.linalg.det(moving.affine[:3, :3]))
    initial.SetScale(float((moving_volume / fixed_volume) ** (1 / 3)))
    reports = []
    for stage in ["similarity", "affine"]:
        if stage == "similarity":
            transform = initial
        else:
            transform = sitk.AffineTransform(3)
            transform.SetCenter(initial.GetCenter())
            transform.SetMatrix(initial.GetMatrix())
            transform.SetTranslation(initial.GetTranslation())
        method = _registration(sitk, fixed_mask, moving_mask, threads, stage)
        method.SetInitialTransform(transform, inPlace=True)
        initial_parameters = list(transform.GetParameters())
        method.Execute(fixed_image, moving_image)
        reports.append(
            {
                "stage": stage,
                "metric_type": "correlation"
                if stage == "affine"
                else "Mattes mutual information (32 bins)",
                "sampling_fraction": 0.15,
                "sampling_seed": 2026,
                "shrink_factors": [8, 4, 2, 1]
                if min(fixed_mask.GetSize()) >= 128
                else [2, 1],
                "smoothing_sigmas_mm": [3, 2, 1, 0]
                if min(fixed_mask.GetSize()) >= 128
                else [1, 0],
                "initial_parameters": initial_parameters,
                "metric": float(method.GetMetricValue()),
                "valid_points": int(method.GetMetricNumberOfValidPoints()),
                "stop": method.GetOptimizerStopConditionDescription(),
            }
        )
    return transform, reports


def resample(image, reference, transform, is_mask=False):
    sitk = sitk_module()
    dtype = np.uint8 if is_mask else np.float32
    source = to_sitk(image, np.asarray(image.dataobj, dtype=dtype))
    target = to_sitk(reference, np.zeros(reference.shape, np.uint8))
    result = sitk.Resample(
        source,
        target,
        transform,
        sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear,
        0,
        sitk.sitkUInt8 if is_mask else sitk.sitkFloat32,
    )
    return array_from_sitk(result)
