"""""
Dimensoes das variaveis utilizadas:
theta  [0, N-1] com N = nº de pontos obtidos pelo lazer
rho [0,N]
c_tr [] : matriz diagonal com 2N linhas e colunas , N correspondentes ao theta e N ao rho
thrsholds=
x  [0,N-1] : valores de x de todos os pontos
y  [0,N-1] : valores de y em todos os pontos
xy  [1, N-1] : valores de x numa linha e de y noutra para todos os pontos

r  [Nlines-1, 0], agora com Nlines = ao numero de segmentos de retas
alpha  [Nlies-1, 0]
pointsidx [Nlines -1,1] : primeiro e ultimo indice dos segmentos de reta
segmends [Nlines, 3] : coordenadas dos pontos iniciais e finais das linhas
segmlen [Nlines, 0] : somprimento dos segmentos de reta

R_seg 
c_trmat 
"""""


def extractlines(theta, rho, c_tr, thersholds):
    # passa de coordenadas polares para cartesianas

    x,y = pol2cart(theta, rho)

    xy = [[x],[y]]

    startidx =0
    endidx = y.shape[1] -1 #x e y são vetores linha
    # faz a extracao das linhas
    alpha, r, pointsidx = splitlines(xy, startidx, endidx, thersholds)

    # numero de segmentos de reta, caso seja mais do que um segmento, vereifica se sao colineares
    n = r.shape[0]
    if n > 1:
        alpha, r, pointsidx = mergecolinear(xy, alpha, r, pointsidx, thersholds)
        n = r.shape[0]
        # atualiza o numero de segmentos

    # definir coordenads dos endpoints e len dos segmentos
    segmends = np.zeros(n - 1, 3)
    segmlen = np.zeros(n - 1, 0)

    for l in range(0, n - 1):
        segmends[l, :] = [np.transpose(xy[:, pointsidx(l, 0)]), np.transpose(xy[:, pointsidx(l, 1)])]
        segmlen[l] = math.sqrt((segmends((l, 0)) - segmends((l, 2))) ** 2 + (segmends((l, 1)) - segmends((l, 3))) ** 2)


    # remover segmentos demasiados pequenos
    #alterar thersholds para params.MIN_SEG_LENGTH e params.MIN_POINTS_PER_SEGMENT
    goodsegmidx = np.argwhere((segmlen >= thersholds.min_seg_length) & ((pointsidx[:,1] - pointsidx[:,0]) >= thersholds.min_points_per_segment ))
    pointsidx = pointsidx[goodsegmidx, :]
    alpha = alpha(goodsegmidx)
    r = r(goodsegmidx)
    segmends = segmends[goodsegmidx, :]
    segmlen = segmlen(goodsegmidx)


    # definiçao de z, R
    z = np.zeros((alpha.shape[0] - 1, r.shape[0] - 1))
    z = ([[alpha], [r]])

    R_seg = np.zeros((1, 1, len([len(alpha), 1])-1))
    n_alpha = alpha.shape[0]

    if c_tr.shape[0] > 0:
        R_seg = np.zeros((1, 1, len([len(alpha), 1])-1))

        for k in range(0, n_alpha - 1):
            aux_range = len(range(pointsidx(k, 0), pointsidx(k, 1)))
            n_pointsegm = len([range, 1])
            c_trmat = [[c_tr(aux_range - 1, aux_range - 1), np.zeros(n_pointsegm - 1)],
                       [np.zeros(n_pointsegm - 1), c_tr(n_alpha + aux_range - 1, n_alpha + aux_range - 1)]]
            R_seg = fitLinePolar(theta(aux_range - 1), rho(aux_range - 1), c_trmat)[2]

    return z, R_seg, segmends
