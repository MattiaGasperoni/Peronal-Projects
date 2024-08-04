from ultralytics import YOLO
import cv2
import numpy as np
import math
from itertools import combinations
from itertools import permutations 

# alrgoritmo finale per la qualità del pezzo

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.1
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.1

# Parametri del testo
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colori
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)

def run_dnn(input_image, net):
	# Create a 4D blob from a frame.
	blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)

	# Sets the input to the network.
	net.setInput(blob)

	# Runs the forward pass to get output of the output layers.
	output_layers = net.getUnconnectedOutLayersNames()
	outputs = net.forward(output_layers)
	# print(outputs[0].shape)

	return outputs

def checkPunto(x, y,img, f=50, NMAX=5):
    a = int(x - (f/2))
    if a >= 0:
        a = int(x - (f/2))
    else:
        a = 0 

    b = int(x + (f/2))
    if b <= img.shape[1]:
        b = int(x + (f/2))
    else:
        b = img.shape[1]

    c = int(y - (f/2))
    if c >= 0:
        c = int(y - (f/2))
    else:
        c = 0
         
    d = int(y + (f/2))
    if d <= img.shape[0]:
        d = int(y + (f/2))
    else:
        d = img.shape[0]

    quads = img[c:d,a:b]
    
    quads_a = cv2.cvtColor(quads, cv2.COLOR_BGR2GRAY)
    nz = np.count_nonzero(quads_a)
    res = False
    if nz > NMAX:
        res = True
    return res

def checkAngolo(x, y, q, w, img, f=50, NMAX=5):
    #primo punto medio
    a = int(x - (f/2))
    if a >= 0:
        a = int(x - (f/2))
    else:
        a = 0 

    b = int(x + (f/2))
    if b <= img.shape[1]:
        b = int(x + (f/2))
    else:
        b = img.shape[1]

    c = int(y - (f/2))
    if c >= 0:
        c = int(y - (f/2))
    else:
        c = 0
         
    d = int(y + (f/2))
    if d <= img.shape[0]:
        d = int(y + (f/2))
    else:
        d = img.shape[0]

    quads = img[c:d,a:b]
    
    #secondo punto medio
    e = int(q - (f/2))
    if e >= 0:
        e = int(q - (f/2))
    else:
        e = 0 

    g = int(q + (f/2))
    if g <= img.shape[1]:
        g = int(q + (f/2))
    else:
        g = img.shape[1]

    h = int(w - (f/2))
    if h >= 0:
        h = int(w - (f/2))
    else:
        h = 0
         
    i = int(w + (f/2))
    if i <= img.shape[0]:
        i = int(w + (f/2))
    else:
        i = img.shape[0]

    quads2 = img[h:i,e:g]

    quads_a = cv2.cvtColor(quads, cv2.COLOR_BGR2GRAY)
    quads_b = cv2.cvtColor(quads2, cv2.COLOR_BGR2GRAY)
    nz = np.count_nonzero(quads_a)
    nc = np.count_nonzero(quads_b)
    res = False
    if nz > NMAX and nc > NMAX:
        res = True
    return res

def post_process(input_image, outputs,maschera):
    class_ids = []
    confidences = []
    boxes = []

    rows = outputs[0].shape[1]

    image_height, image_width = input_image.shape[:2]

    # Resizing factor.
    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT

    # Iterate through 25200 detections.
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]

        # Discard bad detections and continue.
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]

            # Get the index of max class score.
            class_id = np.argmax(classes_scores)

            #  Continue if the class score is above threshold.
            if (classes_scores[class_id] > SCORE_THRESHOLD):
                confidences.append(confidence)
                class_ids.append(class_id)

                cx, cy, w, h = row[0], row[1], row[2], row[3]

                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                box = np.array([left, top, width, height])
                boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    data=[]
    for i in indices:
        b = boxes[i]
        d = {
        "box" : boxes[i],
        "center":(int(b[0]+ b[2]/2),int(b[1]+ b[3]/2)),
        "confidence" : confidences[i],
        }
        data.append(d)
    
    ang=[]
    for d in data:  #Punto centrale dell angolo
        ang.append(d["center"])       
        filtro=FiltroArrey(ang)
        for i in filtro:
            cv2.circle(input_image,(i[0],i[1]),4,BLUE,cv2.FILLED)
           
    indici = range(len(filtro))
    comb = list(combinations(indici,2))
    medio = [0,0]

    for c in comb:          #DISTANZA TRA I PUNTI (ROSSA)
        d1 = data[c[0]]
        d2 = data[c[1]]
        dist = math.dist(d1["center"],d2["center"])
        dist = round(dist,2)

        
        medio[0] = (d1["center"][0] + d2["center"][0])/2
        medio[1] = (d1["center"][1] + d2["center"][1])/2
        if (checkPunto(medio[0],medio[1], maschera)):
            cv2.putText(input_image,"{:.0f}".format(dist), (int(medio[0])-30,int(medio[1])) , FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)  

    comb3 = list(permutations(indici,3))
    som = []
    AngMedio1 = [0,0]
    AngMedio2 = [0,0]
    for m in indici:
        somma1 = 0
        som = []
        c=[x for x in comb3 if x[0]==m]

        for n in c:  #GRADO ANGOLI 
            if n[1] + n[2] in som:
                pass
            else:
                P1 = data[n[0]]
                P2 = data[n[1]]
                P3 = data[n[2]]
                somma1 = n[1] + n[2]  
                som.append(somma1)
                radian = np.arctan2(P2["center"][1]-P1["center"][1],P2["center"][0]-P1["center"][0]) - np.arctan2(P3["center"][1]-P1["center"][1],P3["center"][0]-P1["center"][0])
                ang = np.abs(radian * 180 / np.pi)
                if ang > 180:
                    ang = 360-int(ang)
                ang = round(ang,2)
                

                AngMedio1[0]=(P1["center"][0] + P2["center"][0])/2
                AngMedio1[1]=(P1["center"][1] + P2["center"][1])/2

                AngMedio2[0]=(P1["center"][0] + P3["center"][0])/2
                AngMedio2[1]=(P1["center"][1] + P3["center"][1])/2               
                if (checkAngolo(AngMedio1[0], AngMedio1[1], AngMedio2[0], AngMedio2[1], maschera)):
                    cv2.putText(input_image,"{:.0f}".format(ang),(int(P1["center"][0]),int(P1["center"][1])), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA) 
    return input_image

def FiltroArrey(_in,thresh = 30):
    _out = [_in[0]]
    for i in _in:
        _skip = False
        for o in _out:
            dist = math.dist(i,o)
            if dist<thresh:
                _skip = True
                # print("Skipping",i, "in", _out)
        if not _skip:
            _out.append(i)
    return _out


# caricamento modello Yolov8
segmodel = YOLO("Modelli/v8/Segmentation-StaffeV8.pt")

# caricamento modello Yolov5
modelWeights = "Modelli/v5/detection-angoli-1.onnx"
detectmodel = cv2.dnn.readNet(modelWeights)

Camera = cv2.VideoCapture(0)

while True:
    succes, frame= Camera.read()
    resSegment = segmodel.predict(frame)
    resDetect = run_dnn(frame,detectmodel)

    if (len(resSegment)>0):
        masks = resSegment[0].masks
        if masks and len(masks)>0:
            pippo =  masks.data.numpy()
            m1 = np.any(pippo,axis=0).astype(np.uint8) #conversione true e false in binario 
            m1*=255
            m2 = cv2.bitwise_and(frame,frame, mask = m1) #unione maschere
            m2 = post_process(m2.copy(),resDetect,m2) #disegno angoli su maschera
            cv2.imshow('QualitaPezzo', m2)
       
    
    if cv2.waitKey(1) == ord('q'):
        break
