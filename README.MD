## Yolo-V4-API

### Сетевой сервис для инференса Yolo-V4
### Network service for Yolo-V4 inference

Огромное спасибо https://github.com/philipperemy за его библиотеку python-darknet-yolo-v4, которую я немного переписал под свой сервис.

Big thanks to https://github.com/philipperemy for his python-darknet-yolo-v4, which I changed a little to make it work with my service.

#### Для работы необходим GPU, если хотите CPU, то нужно пересобрать darknet от AlexeyAB.
#### You need GTX 2070 GPU to start libdarknet.so in this repository, or to build your own from AlexeyAB

### Prerequisites
* Сначала установите необходимые зависимости и склонируйте репозиторий.
* First install the requirements and clone the repository.
```bash
sudo apt-get update
sudo apt-get install -y pkg-config git build-essential libopencv-dev wget cmake
git clone https://github.com/AlexeyAB/darknet.git
cd darknet
```
* Change Makefile in accordance with your system configuration and compile the Darknet.
* Внесите необходимые изменения в Makefile в соответствии с конфигурацией вашей системы и скомпилируйте darknet.
````bash
make build
````
* After this put libdarknet.so in src/cfg folder.
* После этого полученный файл libdarknet.so необходимо поместить в папку src/cfg.
* Then, download the weights by following the instructions here: https://github.com/AlexeyAB/darknet (I'm using this one https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT).
* Далее, скачайте веса в соответствии с инструкцией: https://github.com/AlexeyAB/darknet (Используется https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT).
* Put the yolov4.weights to src/cfg path.
* Поместите файл yolov4.weights в папку scr/cfg.
#### Включенная в репозиторий darknet-сеть собрана для:
#### Darknet included in this repository is built with:
* GeForce GTX 2070
* Обязательна CUDA и CUDNN / CUDA and CUDNN required
***
#### Для сборки образа:
#### To build the image:
* Перейти в корневой каталог проекта и ввести команды:
* Go to the project root folder and enter:
````bash
sudo chmod +x ./build_image.sh
sudo ./build_image.sh 
````

#### Для запуска образа вне docker-compose:
#### To start image without docker-compose:
* Перейти в корневой каталог проекта и ввести команды:
* Go to the project root folder and enter:
````bash
sudo chmod +x ./run_image.sh
sudo ./run_image.sh 
````

#### Для запуска в docker-compose:
#### To start the image with docker-compose:
* В корневой папке находится docker-compose.yaml
* There is docker-compose.yaml in the root path:
````yaml
version: "2.3"

services:
  yolov4api:
    image: yolov4_inference_api_gpu:latest
    runtime: nvidia
    environment:
      - "NVIDIA_VISIBLE_DEVICES=0"
    ports:
      - "9999:9999"
````
***
#### Доступные запросы:
#### Available requests:
* /api/classify_small

Классификация с resize входного файла(самый быстрый инференс):

Classification with input file resize (fastest inference):
```bash
curl --location --request POST 'http://0.0.0.0:9999/api/classify_small' \
--header 'Content-Type: image/jpeg' \
--data-binary '@car.jpeg'
```

* /api/classify_full

Классификация полного входного файла:

Full image classification:
````bash
curl --location --request POST 'http://0.0.0.0:9999/api/classify_full' \
--header 'Content-Type: image/jpeg' \
--data-binary '@car.jpeg'
````

* /api/draw_image

Классификатор возвращает входное фото с нарисованными рамками:

Classificator returns input photo with frames on it:
````bash
curl --location --request POST 'http://0.0.0.0:9999/api/draw_image' \
--header 'Content-Type: image/jpeg' \
--data-binary '@car.jpeg'
````
