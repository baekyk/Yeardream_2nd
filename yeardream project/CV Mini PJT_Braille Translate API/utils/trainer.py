from keras.preprocessing.image import ImageDataGenerator

def data_gen():
    images_dir = '../Braille_Dataset'

    datagen = ImageDataGenerator(rotation_range=5,
                                 shear_range=5, 
                                 validation_split=0.2,
                                 ) #20%를 검증모델로 사용.

    train_generator = datagen.flow_from_directory(images_dir,
                                                  target_size=(50,50),
                                                  subset='training')

    val_generator = datagen.flow_from_directory(images_dir,
                                                target_size=(50,50),
                                                subset='validation')

    return train_generator, val_generator

def load_image(img_path):
    images_dir = img_path
    datagen = ImageDataGenerator()
    real_generator = datagen.flow_from_directory(images_dir,
                                                 target_size=(50, 50))

    return real_generator

def train():
    ## data Generator 테스트, 검증 데이터 생성
    train_generator, val_generator = data_gen()

    #MAKE MODEL *한번 모델이 생성되면 다시 실행할 필요 없음
    import mk_model
    hist = mk_model.Make_model(train_generator,val_generator)
    mk_model.print_acc_loss(hist)

