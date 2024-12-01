import LunarAI as ai
model = ai.model.Model()
model.add(ai.layers.layer.Layer(size = 20, inputs = 1))
model.add(ai.layers.layer.Layer(size = 3, inputs = 20))
print(model(ai.ai_libs.array_type.Array([23424234])))
print('\n\n\n\nPress Ctrl+C to quit...')
while True:
    try:
        pass
    except:
        exit()
