import transfer
import Braille_API

# acc확인
# acc = mk_model.acc_chk(model,val_generator)

'''
mode
1 : 'API',
2 : 'braille to string',
3 : 'string to braille'
'''

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
