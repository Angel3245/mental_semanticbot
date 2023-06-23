Aplicación de técnicas de aprendizaje profundo al desarrollo de un bot conversacional para contribuír a la mejora de la salud mental
=======

Proyecto desarrollado en el contexto de la Beca de Colaboración en departamentos universitarios para el curso académico 2022-2023 por el alumno Jose Ángel Pérez Garrido y tutorizado por Dra. Anália Maria Garcia Lourenço del Departamento de Informática de la Universidade de Vigo. 

Esta aplicación proporciona las herramientas necesarias para la creación y evaluación de un bot conversacional basado en la recuperación de la mejor respuesta dentro de un corpus de información para una consulta del usuario mediante la similitud semántica.

La arquitectura del bot está conformada por dos modelos: un modelo Recuperador (BM25) y un modelo Puntuador (SBERT). 


# Requisitos
1. **S.O.** Windows 10/Ubuntu 20.04 o superiores
2. **Intérprete Python 3.** Probado en la versión [3.9](https://www.python.org/downloads/release/python-390/).
3. **Base de Datos ElasticSearch.** Probado en la versión [7.6.2](https://www.elastic.co/es/downloads/past-releases/elasticsearch-7-6-2).
4. **(Opcional) Base de Datos MySQL.** Probado en la versión [10.4.25-MariaDB](https://mariadb.com/kb/en/mariadb-10425-release-notes/).


# Instalación del entorno
Deben seguirse los siguientes pasos para instalar las dependencias necesarias para la ejecución de la aplicación:

1. Desplazarse hasta el directorio del proyecto
```
cd (`ruta_del_proyecto`)
```

2. Crear un entorno virtual (de nombre venv)
```
python3 –m venv venv
```

3. Activar el entorno virtual

Windows: ```venv/bin/activate.bat```

Linux: ```source venv/bin/activate```

4. Instalar las dependencias incluídas en el archivo requirements.txt
```
pip install –r requirements.txt
```


# Uso de la aplicación
Cuando finalice la instalación ya se pueden utilizar los distintos scripts que se encuentran dentro del directorio app y se ejecutan desde la consola de comandos situada na raíz da carpeta do proxecto. El formato del comando sería el siguiente:

```
python3 app/(`nombre_script`).py –o option (`–a args`)
```

Cada script posée un manual propio que se puede consultar con la opción _--help_. Por ejemplo, para _chatbot.py_ el formato del comando sería:

```
python3 app/chatbot.py --help
```


# Scripts
Los scripts disponibles según el tipo de usuario son los seguintes:

## Scripts de usuario

* ```chatbot.py```: contiene la lógica necesaria para la ejecución del bot conversacional. Carga el modelo de _/file/chatbot_model_. Los pesos del modelo entrenado están disponibles en [HuggingFace](https://huggingface.co/Angel3245/PHS-BERT-finetuned-MentalFAQ).

## Scripts de desarrollador

* ```database_scripts.py```: contiene la lógica necesaria para la comunicación con la base de datos.
* ```evaluate.py```: contiene la lógica necesaria para evaluar la eficiencia de los modelos.
* ```generate_training_data.py```: contiene la lógica necesaria para crear los datos de entrenamiento en el formato requerido por el modelo.
* ```prepare_dataset.py```: contiene la lógica necesaria para realizar la limpieza, transformación, construcción y adaptación de los datos para el caso de uso.
* ```test_cuda.py```: contiene la lógica para la comprobación de que CUDA está disponible y configurado correctamente en el equipo.
* ```train.py```: contiene la lógica para entrenar los modelos puntuadores.

# Pasos para la creación y evaluación de un bot conversacional
1. Crear el corpus (MentalFAQ) parseando los conjuntos de datos obtenidos de fuentes externas y añadir los casos de prueba mediante ```prepare_dataset.py``` (ya creado en _/file/data_)

```
python app/prepare_dataset.py -o create_dataset -d MentalFAQ
python app/prepare_dataset.py -o append_test_data -d MentalFAQ -q user_query
```

2. Introducir los datos del corpus en la base de datos ElasticSearch mediante ```database_scripts.py```

```
python app/database_scripts.py -o ingest_db -d MentalFAQ
```

3. Generar los datos de entrenamiento del modelo puntuador mediante ```generate_training_data.py```

```
python app/generate_training_data.py -o generate_hard -d MentalFAQ
python app/generate_training_data.py -o generate_gt -d MentalFAQ
```

4. Entrenar el modelo BERT seleccionado con los mejores hiperparámetros mediante ```train.py```

```
# Obtener un CSV con los resultados de las pruebas
python app\train.py -o hyperparameter_tuning -d MentalFAQ -m bert-base-uncased,mental/mental-bert-base-uncased,mental/mental-roberta-base,publichealthsurveillance/PHS-BERT 

# Entrenar el modelo
python app/train.py -o model_training -d MentalFAQ -m publichealthsurveillance/PHS-BERT -e 2 -b 32 -lr 3e-05
```

5. Generar los resultados con los casos de prueba y calcular las métricas mediante ```evaluate.py```

```
python app/evaluate.py -o generate_topk -d MentalFAQ
python app/evaluate.py -o generate_BERT_results -d MentalFAQ
python app/evaluate.py -o generate_reranked_results -d MentalFAQ
python app/evaluate.py -o evaluation -d MentalFAQ
```

# Estructura del proyecto
El proyecto se organiza de la siguinte manera:

*	_/app_: directorio que almacena los scripts y todo el código de la aplicación organizado en diferentes subdirectorios:
    *   _/chatbot_: contiene las funciones necesarias para devolver una respuesta al usuario a partir de una entrada empleando el sistema de búsqueda semántica.
    *	_/clean_data_: contiene las funciones necesarias para realizar unha limpeza de los datos.
    *	_/DB_: contiene las funciones necesarias para realizar unha conexión con la Base de Datos.
    *	_/model_: contine las entidades para la comunicación con la Base de Datos mediante SQLAlchemy o ElasticSearch.
    *	_/parsers_: continne las clases para analizar, limpiar y extraer datos de diferentes fuentes de información.
    *	_/semantic_search_: contiene las funciones necesarias para entrenar, validar y probar el sistema de búsqueda semántica. (Ml-Recipes, s. f.)
    *	_/shared_: contiene las funciones comunes empleadas por diferentes módulos, como la lectura o escritura de ficheros.
    *	_/transformation_: contiene las funciones necesarias para preparar los conjuntos de datos.
    *	_/view_: contiene las funciones relacionadas con la vista que se presenta al usuario de la aplicación.
*	_/file_: directorio que almacena información requerida para la ejecución de la aplicación organizada en diferentes subdirectorios:
    *	_/chatbot_model_: directorio que almacena los pesos del modelo puntuador (SBERT) seleccionado para el bot conversacional.
    *	_/data_: contiene los conjuntos de datos preparados mediante la aplicación.
    *	_/datasets_: contiene los conjuntos de datos extraídos de diferentes fuentes de datos.
    *	_/test_: contiene los casos de prueba para la evaluación de los modelos.
*	_/output_: directorio donde se almacenan los modelos puntuadores entrenados mediante la aplicación junto con los resultados de su evaluación.


# Referencias
```
Ml-Recipes. (s. f.-b). GitHub - ML-Recipes/BERT-FAQ: Python-based toolkit for building and evaluating a transformer-based FAQ retrieval system. GitHub. https://github.com/ML-Recipes/BERT-FAQ

Gbgonzalez. (s. f.). GitHub - gbgonzalez/reddit_extraction. GitHub. https://github.com/gbgonzalez/reddit_extraction
```
