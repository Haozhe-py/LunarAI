import tensorflow as tf
from LunarAI.transformer import Transformer, create_padding_mask, create_look_ahead_mask, create_decoder_mask

# 定义损失函数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))  # 忽略填充部分
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

# 定义优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

@tf.function
def train_step(inp, tar, model:Transformer):
    tar_inp = tar[:, :-1]  # 目标输入（去掉最后一个词）
    tar_real = tar[:, 1:]  # 目标真实值（去掉第一个词）

    # 创建掩码
    enc_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tar_inp.shape[1])
    dec_padding_mask = create_decoder_mask(inp)

    with tf.GradientTape() as tape:
        predictions, _ = model([inp, tar_inp, enc_padding_mask, look_ahead_mask, dec_padding_mask], training=True)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    return loss

def fit(model:Transformer, epochs:int, inp_data, tar_data):
    for epoch in range(epochs):
        total_loss = 0
        for batch, (inp, tar) in enumerate(zip(inp_data, tar_data)):
            batch_loss = train_step(inp, tar)
            total_loss += batch_loss
            if batch % 10 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {batch_loss.numpy()}')
        print(f'Epoch {epoch + 1} Loss {total_loss / len(inp_data)}')
