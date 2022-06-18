import sys

from cmath import pi
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import scipy.linalg
import statistics
"""
1ºteste:  tudo constante
"""
valor_medio_x = np.array([0.012188232293023637, 0.014016202813612369, 0.014039838118958235, 0.013630260222076876, 0.012904320042201465, 0.012351735186264869, 0.01365916051244085, 0.012004254110014509, 0.012346839719211617, 0.013860186582124652, 0.01306617177927651, 0.012586365906184203, 0.013508160860147018, 0.013085701870422666, 0.015489879478934307, 0.013246922677629246, 0.014963170697131904, 0.014545784390561697, 0.012138071407465081, 0.013120606794432457, 0.013340802647808254, 0.012056104250265981, 0.01339065160628886, 0.01381185277385561, 0.01156673942348371, 0.012536246094256352, 0.02211188733422123, 0.011106263093796528, 0.012978093558603038, 0.01595323056926915, 0.01348174826618734, 0.011826838976268761, 0.012453704997240413, 0.012668378605957925, 0.011807621343624213, 0.014990579426447763])
valor_medio_y = np.array([0.1886048910291877, 0.04948684217646324, 0.03535118587321408, 0.05390578767715626, 0.0435575737558866, 0.1430559531948646, 0.10894705719528237, 0.11028453741215964, 0.07698010339260468, 0.188288237678517, 0.05024473727306478, 0.046646823698632635, 0.09083160084167516, 0.09401734919140052, 0.09346442161130177, 0.11874718251378852, 0.12321509746083782, 0.07763124646112504, 0.07358362279202212, 0.18357341699369023, 0.0487630025594579, 0.10821924080672624, 0.1218499744799659, 0.03921871705147275, 0.08854079116886616, 0.04593319919463616, 0.1907004811989628, 0.0654833058280961, 0.15081326040406287, 0.11852555692149715, 0.11671318418283616, 0.0628552177323286, 0.075970484529678, 0.10754175261302526, 0.048590817675034724, 0.070716520035896])
valor_medio_theta = np.array([0.015070312727453907, 0.020878610086600975, 0.018502169038143133, 0.020260918453239348, 0.020410750033957852, 0.012648768016157568, 0.019836883109103566, 0.014887425831877096, 0.018682142362236098, 0.020591282223646326, 0.018402598966713885, 0.015171864297761996, 0.020355006492085766, 0.019590749178084427, 0.020478426872811054, 0.019766904315720982, 0.019773205370721678, 0.019825492813108556, 0.012831415710576497, 0.019985291249489008, 0.020070500096800604, 0.015182413286418497, 0.01942275524745421, 0.02016641357170971, 0.012422347297701497, 0.016221867734066676, 0.029507780260644528, 0.013488573556757436, 0.018774984339700953, 0.023472376969072728, 0.019843685491544343, 0.015680209771552357, 0.017667088750906276, 0.016911738423969272, 0.012564846000233826, 0.019559815459007705])
max1x = np.array([0.11692810380674601, 0.10553585746082628, 0.1157340394515669, 0.11147710539828581, 0.11766709260951724, 0.1124688298813804, 0.1000651428590218, 0.11525831283616628, 0.12829307691782932, 0.12254684237719893, 0.12008189707870165, 0.11664747879583448, 0.10670318488485897, 0.11592912648609466, 0.11243117715730055, 0.11380685334530938, 0.10994638609646257, 0.10710859595151234, 0.11319398333189512, 0.11027346721579379, 0.11546757694228882, 0.11247420930639584, 0.11200685377539887, 0.12653192576586758, 0.12090059635844574, 0.11286524039927004, 0.14315424466985316, 0.11145975009732245, 0.12057855035798137, 0.12430403776473631, 0.11582497816071635, 0.10041049827719939, 0.11625973936544443, 0.10018191906213436, 0.10808144006244746, 0.11551955273266423])
max1y = np.array([0.48528807719157196, 0.12044685825784684, 0.10071499500639147, 0.16161889996324552, 0.16125879530231302, 0.31630269728862936, 0.2470131193940901, 0.24042014608750462, 0.19780683451693415, 0.37790261206026177, 0.14413286659573643, 0.14645522093771923, 0.24496709882238754, 0.2238565246289408, 0.2508283029793241, 0.27497407391484985, 0.30400589185954807, 0.20282159778456466, 0.2052012798720246, 0.42280748591078243, 0.1377475811572244, 0.3084335584217053, 0.25955134189013407, 0.11497622683002717, 0.2615861720397037, 0.12157481927575553, 0.39084763898616837, 0.16657623760199547, 0.3510236201305732, 0.2383071835366226, 0.26609096906664376, 0.15704869989329762, 0.21035462053618303, 0.2569859586373188, 0.20555906730698492, 0.20241469587710492])
max1t = np.array([0.12417293934363904, 0.14254542229126588, 0.1370482706913596, 0.1373636214803089, 0.14242670876258523, 0.10728438725729639, 0.13602556751223416, 0.1334552355538896, 0.1380911086599892, 0.14107172987207095, 0.13599156067278262, 0.12646335318713997, 0.14001724304629537, 0.14106427348012662, 0.141241319140059, 0.13515001858547993, 0.13929772727692158, 0.13628291133914705, 0.1090679368327987, 0.13876046472278292, 0.13841736226816193, 0.12588675048352505, 0.13250749205209655, 0.1365779372943723, 0.11110241055814396, 0.12401719948230583, 0.1443791569570534, 0.11206852045374927, 0.12581418093148633, 0.14890421766149364, 0.1353424838725914, 0.12520305381414132, 0.13826366106590493, 0.13428882586458935, 0.1046354036867958, 0.1355767263240708])

mean_medx = statistics.mean(valor_medio_x)
print(mean_medx)

mean_medy = statistics.mean(valor_medio_y)
print(mean_medy)

mean_medtheta = statistics.mean(valor_medio_theta)
print(mean_medtheta)

mean_max1x = statistics.mean(max1x)
print(mean_max1x)

mean_max1y = statistics.mean(max1y)
print(mean_max1y)

mean_max1t = statistics.mean(max1t)
print(mean_max1t)



"""
2ºteste: variar ruído, manter o resto constante
"""
#valores constantes:

valor_ruido = np.array([])
valor_medio_xr = np.array([])
valor_medio_yr = np.array([])
valor_medio_thetar = np.array([])



#colocar no powerpoint o valor das covaraincia a dividir pelo numero de testes


"""
3ºteste: variar g
"""

valor_medio_xg = np.array([])
valor_medio_yg = np.array([])
valor_medio_thetag = np.array([])



"""
3ºteste: linha-> ver a covariancia da linha , mahalanobis distance ??
"""


