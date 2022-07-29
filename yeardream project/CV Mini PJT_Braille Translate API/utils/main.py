import transfer
import Braille_API
import tensorflow as tf

# acc확인
# acc = mk_model.acc_chk(model,val_generator)

'''
mode
1 : 'API',
2 : 'braille to string',
3 : 'string to braille'
'''


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

mode = 'API'


if mode == 'API':
    Braille_API.flask()
elif mode == 'braille to string':
    path = './static/people.png'
    transfer.img_to_s(path)
elif mode == 'string to braille':
    transfer.s_to_img()
else:
    print('Please re-choice mode')
    quit()


# import trainer
# with tf.device("/device:GPU:0"):
# 
#     trainer.train()
