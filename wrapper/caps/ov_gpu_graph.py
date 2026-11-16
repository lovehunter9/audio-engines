"""OpenVINO GPU graph compilation with a shared device context."""
import logging
from functools import lru_cache


@lru_cache(maxsize=1)
def _runtime():
    import openvino as ov
    core = ov.Core()
    return core, core.get_default_context("GPU")


def promote_1d_convolutions(model):
    """Use equivalent 2D kernels for the GPU plugin's 1D convolution lowering."""
    import numpy as np
    from openvino import opset13 as ops
    count = 0
    for node in model.get_ordered_ops():
        kind = node.get_type_name()
        if kind not in ('Convolution', 'GroupConvolution', 'ConvolutionBackpropData', 'GroupConvolutionBackpropData'):
            continue
        if node.input_value(0).get_partial_shape().rank.get_length() != 3:
            continue
        attributes = node.get_attributes()
        data = ops.unsqueeze(node.input_value(0), np.array([2], np.int64))
        weights_axis = 3 if kind.startswith('Group') else 2
        weights = ops.unsqueeze(node.input_value(1), np.array([weights_axis], np.int64))
        options = dict(strides=[1, *attributes['strides']],
                       pads_begin=[0, *attributes['pads_begin']],
                       pads_end=[0, *attributes['pads_end']],
                       dilations=[1, *attributes['dilations']],
                       auto_pad=attributes['auto_pad'].upper())
        if 'Backprop' in kind:
            options['output_padding'] = [0, *attributes['output_padding']]
            if node.get_input_size() == 3:
                options['output_shape'] = ops.concat([np.array([1], np.int64), node.input_value(2)], 0)
        function = {'Convolution': ops.convolution, 'GroupConvolution': ops.group_convolution,
                    'ConvolutionBackpropData': ops.convolution_backprop_data,
                    'GroupConvolutionBackpropData': ops.group_convolution_backprop_data}[kind]
        convolution = function(data, weights, **options)
        convolution.set_friendly_name(node.get_friendly_name() + '/spatial_2d')
        restored = ops.squeeze(convolution, np.array([2], np.int64))
        restored.set_friendly_name(node.get_friendly_name())
        restored.output(0).set_names(node.output(0).get_names())
        node.output(0).replace(restored.output(0))
        count += 1
    model.validate_nodes_and_infer_types()
    return count


def compile_gpu(path, *, precision="f32", input_shapes=None):
    core, context = _runtime()
    model = core.read_model(str(path))
    if input_shapes is not None:
        model.reshape({model.input(index): shape for index, shape in input_shapes.items()})
    count = promote_1d_convolutions(model)
    log = logging.getLogger("openvino-gpu")
    log.info("Compiling GPU graph %s (%s equivalent 2D convolutions)", path, count)
    compiled = core.compile_model(model, context, {"INFERENCE_PRECISION_HINT": precision})
    devices = [str(device) for device in compiled.get_property("EXECUTION_DEVICES")]
    if not devices or any(not device.startswith("GPU") for device in devices):
        raise RuntimeError("Neural execution must use OpenVINO GPU: " + repr(devices))
    log.info("Compiled GPU graph %s", path)
    return compiled
