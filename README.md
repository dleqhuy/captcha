# Captcha Convolutional Recurrent Neural Network

## Environment
```bash
$ conda env create -f captcha/captcha.yml
$ conda activate captcha
```

## Data prepare
```bash
$ curl -LO https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip 
$ unzip -qq captcha_images_v2.zip
$ python captcha/crnn/prepare.py \
--config captcha/configs/captcha.yml \
--dir captcha_images_v2
```

## Train
```bash
$ python captcha/crnn/train.py \
--config captcha/configs/captcha.yml \
--save_dir captcha_exp1
```

## Demo
```bash
$ python captcha/crnn/predict.py  \
--config captcha/configs/captcha.yml  \
--weight captcha_exp1/0_best_model.h5 \
--images captcha/captcha_test  \
--post greedy
```