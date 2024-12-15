import LunarAI as ai
from random import randint

model = ai.model.Model()
model.add(ai.layers.layer.Layer(size = 20, inputs = 1))
model.add(ai.layers.layer.Layer(size = 3, inputs = 20))
model.add(ai.layers.layer.Layer(size = 1, inputs = 3, activation = ai.sigmoid))
#print(model(ai.ai_libs.array_type.Array([[23424234]])))

def get(n):
    x_train, y_train = [],[]
    for i in range(n):
        x = randint(-100,100)
        x_train.append([x])
        if (x >= 10) or (x <= -10):
            y = 0
        else:
            y = 1
        y_train.append([y])
        continue
    return ai.ai_libs.array_type.Array(x_train) , ai.ai_libs.array_type.Array(y_train)

xt,yt = get(300)

sgd = ai.train.sgd.SGD()
for i in range(1000):
    j = randint(0,299)
    x,y = ai.ai_libs.array_type.Array([xt[j]]),ai.ai_libs.array_type.Array([yt[j]])
    sgd(
        x,
        y,
        model
    )


print('\n\n\n\nPress Ctrl+C to quit...')
while True:
    try:
        pass
    except:
        exit()
