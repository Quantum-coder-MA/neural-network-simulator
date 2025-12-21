import tensorflow as tf

def build_tracer_model(model):
    tr_out = []
    tr_info = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            tr_out.append(layer.output)
            tr_info.append({
                "layer_name": layer.name,
                "type": "conv",
                "value": "z"
            })

        elif isinstance(layer, tf.keras.layers.Dense):
            tr_out.append(layer.output)
            tr_info.append({
                "layer_name": layer.name,
                "type": "dense",
                "value": "z"
            })

        elif isinstance(layer, tf.keras.layers.Activation):
            tr_out.append(layer.output)
            tr_info.append({
                "layer_name": layer.name,
                "type": "activation",
                "value": "a"
            })

    trace_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=tr_out
    )

    return trace_model, tr_info
