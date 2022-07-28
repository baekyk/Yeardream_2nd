from PIL import Image
import crop
import Predict
import trainer
import mk_model


def img_to_s(path):
    # BrailleNet에 저장된 모델을 불러옴.
    model = mk_model.load_model()

    # 사진 데이터 불러오기, 예측
    Predict.chk_trans()
    pred = Predict.Predic()
    cropped = crop.img_devide(path)

    cropped.create_dir()
    cropped.set_image()
    pred.reset()

    for i in range(0,cropped.lengh):
        cropped.devide_img()
        real = trainer.load_image('../temp')
        pred.Predict(model,real)
        cropped.remove_file()


    result = ''.join(pred.result)
    print(result)
    return result



def s_to_img(word=None):
    image_dir = '../Braille_origin/'

    if word == None:
        word = input()
    
    image_dic = dict()
    for i, c in enumerate(word):
        if c == ' ':
            image_dic[f'image{i}'] = Image.open(image_dir+'zz.png')
        else:
            image_dic[f'image{i}'] = Image.open(image_dir+f'{c}.png')

            
    image_size = image_dic['image0'].size
    new_image = Image.new('RGB',(len(word)*image_size[0], image_size[1]), (250,250,250))

    for i, key in enumerate(image_dic.keys()):
        new_image.paste(image_dic[key],(i*image_size[0],0))
    new_image.save(f'static/{word}.png')
    return new_image



