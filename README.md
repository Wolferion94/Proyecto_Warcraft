# Modelo de clasificación jugadores World of Warcraft - Campos de Batalla
## Contexto

[![Grito de Guerra](https://static.wikia.nocookie.net/es_wowpedia/images/7/73/Warsong_Gulch_loading_screen.jpg/revision/latest?cb=20120826205908 "Grito de Guerra")](http://https://static.wikia.nocookie.net/es_wowpedia/images/7/73/Warsong_Gulch_loading_screen.jpg/revision/latest?cb=20120826205908 "Grito de Guerra")

La información obtenida se trae desde Kagle, específicamente del segmento de videojuegos por medio del siguiente link:

https://acortar.link/QQTMyh

En esta base de datos se evidencian 14650 registros correspondientes a los enfrentamientos dados en los campos de batalla de World of Warcraft, se encuentran disponibles cinco archivos que contienen estadísticas obtenidas al finalizar cada campo de batalla del videojuego World of Warcraft. Estos archivos comparten algunas estadísticas en común. World of Warcraft (abreviado como WoW) es un videojuego de rol multijugador masivo en línea desarrollado por Blizzard Entertainment. Es el cuarto juego lanzado establecido en el universo fantástico de Warcraft, el cual fue introducido por primera vez por Warcraft: Orcs Humans en 1994. 

A continuación se indica en qué consiste el juego y el contexto de los diferentes datos que podrían presentarse, esto es, dentro del juego, existen unos escenarios denominados ”campos de batalla”, en los que los jugadores pertenecientes a las dos facciones jugables (Horda y Alianza) luchan entre sí para conseguir la victoria. Cada campo de batalla tiene sus propias características, que pueden ir desde la captura de territorios hasta la obtención de recursos. Además de proporcionar diversión a los jugadores, participar en los campos de batalla tiene como recompensa la obtención de una moneda en el juego llamada ”Honor”. Esta moneda puede ser intercambiada por objetos de gran poder para los personajes de los jugadores.



# Descripción de la base de datos

La información encontrada en la base de datos del juego World of Warcraft contiene la siguiente información:

Inicialmente información relacionada con características de los jugadores y del campo de batalla y algunas variables relacionadas con el rendimiento del juego medido en los diferentes enfrentamientos.

* Código: Código para el campo de batalla
* Facción: Facción del jugador (Horda o Alianza)
* Clase: Clase del jugador 
>* Warrior: Guerrero
>* Paladin: Paladín
>* Hunter: Cazador
>* Rogue: Pícaro
>* Priest: Sacerdote
>* Death knight: Caballero de la muerte
>* Shaman: Chamán
>* Mage: Mago
>* Warlock: Brujo
>* Monk: Monje
>* Druid: Druida
>* Demon hunter: Cazador de demonios

* KB: Número de muertes mortales dadas por el jugador
* D: Número de veces que murió el jugador
* HK: Número de asesinatos en los que contribuyó el jugador o su grupo
* DD: Daño hecho por el jugador
* HD: Curación realizada por el jugador
* Honor: Honor otorgado al jugador
* Ganar: 1 si el jugador ganó
* Perder: 1 si el jugador perdió
* Rol: ‘dps’ si el jugador es un distribuidor de daños; ‘heal’ si el jugador se concentra en curar a los aliados. Es importante tener en cuenta que no todas las clases pueden ser curanderos, solo chamanes, paladines, sacerdotes, monjes y druidas, pero todas las clases sí pueden causar daño
* BE: Algunas semanas hay un evento de bonificación, cuando aumenta el honor ganado. 1 si el campo de batalla ocurrió durante esa semana

*Battleground*: Representa el tipo de campo de batalla:

* AB: Arathi basinBG: Battle for Gilneas
* DG: Deepwind gorge
* ES: Eye of the storm
* SA: Strand of the ancients
* SM: Silvershard mines
* SS: Seething shore
* TK: Temple of Kotmogu
* TP: Twin peaks
* WG: Warsong gulch

*Descripción de los archivos*

Se tienen 4 archivos que tienen estadísticas para un solo tipo de bg y tienen columnas adicionales:
* wowgil.csv: Estadísticas del campo de batalla de Battle for Gilneas. BA: número de bases asaltadas por el jugador. BD: número de bases defendidas por el jugador.
* wowsm.csv: Estadísticas del campo de batalla de las minas Silvershard. CC: número de carros controlados por el jugador.
* wowtk.csv: Estadísticas del campo de batalla del Templo de Kotmogu: OP: número de orbes controlados por el jugador. VP: puntos de victoria ganados por el jugador.
wowwg.csv: Estadísticas del campo de batalla de Warsong Gulch: FC: número de banderas capturadas por el jugador. FR: número de banderas defendidas por el jugador.

Con este conjunto de datos, se puede verificar qué clase es la más letal, cuál muere con más frecuencia, qué facción gana más escenarios, cómo la captura de una bandera puede afectar el honor otorgado, qué clase es la mejor para hacer daño o curar, qué clases tienen una mejor combinación para ganar cada campo de batalla y cuál es la probabilidad de ganar para cada clase.

*Problema*

De acuerdo a la información descrita anteriormente, el problema que se quiere solucionar en este proyecto es: La clasificación de los jugadores presentes en la base de datos, en tres categorías a partir de la clase del personaje, raza y tipo con respecto a las victorias en dichos escenarios. Para llegar a la solución de este problema se plantea realizar una limpieza y normalización de los datos acompañado de un análisis descriptivo e inferencial de los datos para finalmente realizar el entrenamiento de un modelo de regresión logístico por medio de scikit-learn con el fin de poder hacer la clasificación de los jugadores presentes en la data. 

*Objetivos*

*Objetivo General*
Crear un modelo de regresión logística en Python que permita clasificar el desempeño de los jugadores de acuerdo a su clase en los campos de batalla de World Of Warcraft.

*Objetivos Específicos*
Extraer, limpiar y  normalizar la data “World of Warcraft Battlegrounds” obtenida de Kaggle, utilizando métodos de Python y los conocimientos adquiridos en la clase de Fundamentos de estadística
Realizar el análisis descriptivo de la data “World of Warcraft Battlegrounds” usando los datos limpios y normalizados, con el fin de identificar la clase más letal y el desempeño de esta con respecto a su rol.
Estimar el impacto en el resultado del campo de batalla dado un bonus de honor
Desarrollar un modelo de regresión logística con ayuda de Sckit Learn que permita clasificar si un jugador es bueno, intermedio o malo.

Los cuatro pasos de minería de datos que se van a aplicar a la base de datos son:
* Establecer objetivos
* Preparación de los datos
* Aplicación de algoritmos de minería de datos
* Evaluación de resultados

# Ejecución del modelo

Se debe tener en cuenta que para la ejecución del modelo deben estar en la misma carpeta los archivos .py y el notebook .ipynb.
