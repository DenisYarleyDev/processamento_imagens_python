import cv2 as cv
import numpy as np

#carregar imagem original
imagem = cv.imread('gato2.jpg')
imagem2 = cv.imread('gato3.jpg')


#exibir a imagem original
cv.imshow('original',imagem)
# cv.imshow('original',imagem2)

#aplica filtro cinza
imagemCinza = cv.cvtColor(imagem,cv.COLOR_BGR2GRAY)

#converte a imagem para a escala de cinza
cv.imshow('fitro cinza', imagemCinza)

#aplica o filtro de bordas Canny
bordas = cv.Canny(imagemCinza,100,200)

#exibe a imagem com filtro Canny
cv.imshow('filtro de bordas', bordas)

#aplica o filtro de desfoque gaussiano
imagemBlur = cv.GaussianBlur(imagem,(5,5), 0)

#exibe a imagem com filtro blur gaussiano
cv.imshow('filtro de blur', imagemBlur)

#------------------------------------------
#rotacionar a imagem
(h,w) = imagem.shape[:2]
centro = (w // 2, h // 2)

angulo = 45

mRotacao = cv.getRotationMatrix2D(centro,angulo, 1.0)

#aplicar a rotacao
imagem_rotacao = cv.warpAffine(imagem, mRotacao, (w,h))
#exibir a imagem rotacionada
cv.imshow('rotacao', imagem_rotacao)

#aplicando segmentação (binarizar imagem)
_, imagem_binaria = cv.threshold(bordas, 127,255, cv.THRESH_BINARY)

#exibe a imagem binarizada
cv.imshow('segmento', imagem_binaria)

#aplicando a dilatacação
kernel = np.ones((3,3), np.uint8)
dilatacao = cv.dilate(imagem_binaria, kernel, iterations=18)

#exibindo a dilatacao
cv.imshow('dilatacao',dilatacao)

#soma de imagens
#igualar os tamanhos das imagens
imagem2 = cv.resize(imagem2, (imagem.shape[1], imagem.shape[0]))

#soma das imagens
soma = cv.add(imagem, imagem2)

#exibir imagens somadas
cv.imshow('soma', soma)

#aplicando analise de forma
contornos, _ = cv.findContours(dilatacao, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for contorno in contornos:
    area = cv.contourArea(contorno)
    perimetro = cv.arcLength(contorno, True)

#mostrar o contorno na imagem original
cv.drawContours(imagem, [contorno], -1, (0,255,0), 2)
cv.imshow('contornos', imagem)


#------------------------------------------
#aplicar detecção de cantos método Harris
#converter imagem para 32-bits
imgConvertendo = np.float32(imagemCinza)

#aplicando o metodo Harris
cantos = cv.cornerHarris(imgConvertendo, 2, 5, 0.07)

#marcando os resultados dos cantos dilatados
cantos = cv.dilate(cantos,None)

#revertendo o resultado para a imagem original
imagem[cantos > 0.01 * cantos.max()]=[0,0,255]

#exibindo a imagem com os cantos marcados
cv.imshow('cantos',imagem)

#-------------------------------------------
#salvar a imagem processada
# cv.imwrite('resultado_final.jpg', imagem)


cv.waitKey(0)
cv.destroyAllWindows()