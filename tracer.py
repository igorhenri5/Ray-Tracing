from PIL import Image
import numpy as np
import numbers
import sys
import time
import math 
import random

class vec3:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z


    def __getitem__(self, i):
        if i == 0:
            return self.x

        if i == 1:
            return self.y

        if i == 2:
            return self.z

    # sum vectors
    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    # subtract vectors
    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    # multiply vectors
    def __mul__(self, other):
        if isinstance(other, numbers.Real):
            return vec3(self.x * other, self.y * other, self.z * other)

        return vec3(self.x * other.x, self.y * other.y, self.z * other.z)

    # divide vector by scalar or other vector
    def __truediv__(self, other):
        if isinstance(other, numbers.Real):
            return vec3(self.x / other, self.y / other, self.z / other)

        return vec3(self.x / other.x, self.y / other.y, self.z / other.z)

    # k * u
    def __rmul__(self, other):
        return vec3(other * self.x, other * self.y, other * self.z)

    # vector magnitude
    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    # squared length of vec
    def squared_length(self):
        return self.x * self.x + self.y * self.y + self.z * self.z

    # dot product
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    # cross product
    def cross(self, other):
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x

        return vec3(x, y, z)

    # unit vector
    def unit(self):
        return self / self.length()

    # positive vec3
    def __pos__(self):
        return self

    # negative vec3
    def __neg__(self):
        return vec3(-self.x, -self.y, -self.z)

    def set_(self, v):
        self.x = v.x
        self.y = v.y
        self.z = v.z

    def __repr__(self):
        return '%s(%.3f, %.3f, %.3f)' % (__class__.__name__, self.x, self.y, self.z)

def sphereIntersection(obj, origin, dir, endPos = None, inside = None):
    e = (origin - obj['center'])
    a = dir.dot(dir)
    b = 2 * e.dot(dir)
    c = e.dot(e) - (obj['radius'])**2

    discriminant = b**2 - 4*a*c
    if discriminant <= 0:
        return -1, inside
    discriminant = math.sqrt(discriminant)
    t1 = (-b - discriminant)/(2*a)
    t2 = (-b + discriminant)/(2*a)

    if endPos:
        if dir.x == 0:
            maxT = sys.float_info.max
        else:
            maxT = (endPos.x - origin.x) / dir.x
    else:
        maxT = sys.float_info.max
    if(t1 > 0 and t1 < maxT):
        if(inside != None):
            inside = False
        return t1, inside
    elif(t2 > 0 and t2 < maxT):
        if(inside != None):
            inside = True
        return t2, inside
    else:
        return -1, inside

def polyhedronIntersection(obj, origin, dir, endPos = None):
    t0 = 0 
    t1 = sys.float_info.max
    normal = vec3(0,0,0)

    for plane in obj['planes']:
        n = vec3(plane['a'],plane['b'], plane['c'])
        dn = dir.dot(n)
        val = origin.dot(n) + plane['d'] 

        if(dn <= 0 and dn >= -0):
            if(val > 0):
                t1 = -1
        
        if(dn > 0):
            t = -val / dn
            if(t < t1):
                t1 = t
                nT1 = n            
        
        if(dn < -0):
            t = -val / dn
            if(t > t0):
                t0 = t
                nT0 = n

    if endPos:
        if dir.x == 0:
            maxT = sys.float_info.max
        else:
            maxT = (endPos.x - origin.x) / dir.x
    else:
        maxT = sys.float_info.max

    if(t1 < t0):
        return -1, normal

    if(abs(t0) <= 0 and (t1 >= t0) and t1 < sys.float_info.max):
        normal = (nT1 * (-1)).unit()
        if(t1 < maxT):
            return t1, normal
        else:
            return -1, normal

    if(t0 > 0 and t1 >= t0):
        normal = nT0.unit()
        if(t0 < maxT):
            return t0, normal
        else:
            return -1, normal

    return -1, normal


def intersection(scene, origin, dir, endPos = None, exclObj = None, inside = None):
    closestObj = None
    closestObjT = sys.float_info.max
    closestN = vec3(0,0,0)
    closestInside = False

    for o in scene['objects']:
        if o == exclObj:
            continue
        if o['type'] == "sphere":
            t,i = sphereIntersection(o, origin, dir, endPos, False)
            if t > 0 and t < closestObjT:
                closestObj = o
                closestObjT = t
                closestInside = i
        elif o['type'] == "polyhedron":
            t,n = polyhedronIntersection(o, origin, dir, endPos)
            if t > 0 and t < closestObjT:
                closestN = n
                closestObjT = t
                # pass
                # closestObj= o

    if closestObj:
        intersectedObj = closestObj
        intersection = origin + closestObjT * dir
        if scene['pigments'][intersectedObj['pigment']]['type'] == "solid":
            intersectionColor =  scene['pigments'][intersectedObj['pigment']]['rgb']
        elif (scene['pigments'][intersectedObj['pigment']]['type'] == "checker"):
            val = math.floor(intersection.x / scene['pigments'][intersectedObj['pigment']]['size'] + math.floor(intersection.y / scene['pigments'][intersectedObj['pigment']]['size'] + math.floor(intersection.z / scene['pigments'][intersectedObj['pigment']]['size'])))%2
            if(not val):
                return scene['pigments'][intersectedObj['pigment']]['rgb0']
            else:
                return scene['pigments'][intersectedObj['pigment']]['rgb1']

        elif scene['pigments'][intersectedObj['pigment']]['type'] == "texmap":    
            print("to do")

        if intersectedObj['type'] == "sphere":
            normal = (intersection - intersectedObj['center']).unit()
            if inside:
                inside = False
            if closestInside:
                normal = normal * (-1)
                if inside:
                    inside = True    
        if intersectedObj['type'] == "polyhedron":
            normal = closestN
            # inside = False
            # return False, vec3(0,0,0), vec3(0,0,0), None, False, vec3(0,0,0)

        return True, intersection, intersectionColor, intersectedObj, inside, normal
    return False, vec3(0,0,0), vec3(0,0,0), None, inside, vec3(0,0,0)

def reflectionDirection(dir, normal):
    c = normal.dot(-dir)
    return (dir + (2 * normal * c)).unit()

def transmissionDirection(refrRate, dir, normal):
    invDir = -dir
    c1 = invDir.dot(normal)
    c2 = 1 - refrRate**2 * (1-c1**2)

    if(c2 < 0):
        v = normal * 2 * c1
        transDir = v - invDir
        return True, transDir
    elif c2 > 0:
        transDir =  (normal * (refrRate * c1 - c2)) + (dir * refrRate)
        return True, transDir
    else: 
        return False, None

def rayTracer(scene, p, dir, maxDepth, exclObj = None):

    intersected, q, color, obj, inside, normal = intersection(scene, p, dir, None, None, False)
    if not intersected:
        return scene['lights'][0]['rgb']
    
    ambientColor = scene['lights'][0]['rgb'] * scene['finishes'][obj['finish']]['ka']
    diffuseColor = vec3(0.0, 0.0, 0.0)
    specularColor = vec3(0.0, 0.0, 0.0)

    lights = iter(scene['lights'])
    next(lights)
    for l in lights:
        lightDir = (l['pos'] - q).unit()
        dist = math.sqrt((l['pos'].x - q.x)**2 + (l['pos'].y - q.y)**2 + (l['pos'].z - q.z)**2)
        att = 1/(l['at1'] + dist * l['at2'] + dist**2 * l['at3'])

        intersected, fq, fcolor, fobj, finside, fnormal = intersection(scene, q, lightDir, l['pos'], obj, None)

        if not intersected:
            cosDiff = lightDir.dot(normal)
            if(cosDiff <= 0): 
                cosDiff = 0           
            diffuseColor += l['rgb'] * cosDiff * scene['finishes'][obj['finish']]['kd'] * att

            e = -dir
            halfway = (lightDir + e).unit()
            cosSpec = halfway.dot(normal.unit())
            if(cosSpec <= 0): 
                cosSpec = 0
            cosSpec = (cosSpec)**scene['finishes'][obj['finish']]['a']
            specularColor += l['rgb'] * cosSpec * scene['finishes'][obj['finish']]['ks'] * att
    
    reflectionColor = vec3(0,0,0)
    transmissionColor = vec3(0,0,0)

    if maxDepth>0:
        reflDir = reflectionDirection(dir, normal)
        reflectionColor = rayTracer(scene, q, reflDir, maxDepth - 1, obj) * scene['finishes'][obj['finish']]['kr'] 
        if scene['finishes'][obj['finish']]['ior'] == 0:
            refrRate = math.sqrt(sys.float_info.max/2)
        else:
            refrRate = 1/scene['finishes'][obj['finish']]['ior']
        if inside:
            refrRate = scene['finishes'][obj['finish']]['ior']
        transmit, transDir = transmissionDirection(refrRate, dir, normal)
        if transmit:
            transmissionColor = rayTracer(scene, q, transDir, maxDepth - 1, obj)* scene['finishes'][obj['finish']]['kt']

    finalColor = ambientColor + diffuseColor

    return ((color * finalColor + reflectionColor + transmissionColor + specularColor))
    # return vec3(160,0,160)

def computeScene(height, width, scene, outputFile):
    random.seed(time.time())
    start_time = time.time()

    cameraDir = (scene['camera']['eye'] - scene['camera']['center']).unit()
    d = math.sqrt((scene['camera']['eye'].x - scene['camera']['center'].x)**2 + (scene['camera']['eye'].y - scene['camera']['center'].y)**2 + (scene['camera']['eye'].z - scene['camera']['center'].z)**2)

    aspectHeight = 2*math.tan(math.radians(scene['camera']['fov']/2))*d
    aspectWidth = (aspectHeight * width)/height

    cameraRight = cameraDir.cross(scene['camera']['up'])          
    topLeft  = scene['camera']['eye']
    topLeft -= cameraRight * (aspectWidth/2)
    topLeft += scene['camera']['up'] * (aspectHeight/2)

    pixelRight = (aspectWidth / width) * cameraRight
    pixelBottom = (aspectHeight / height) * (-scene['camera']['up']) 

    with open(outputFile, 'w') as f:
        f.write(f"P3\n{width} {height}\n255\n")
        for h in range(0, height):
            print(h)
            for w in range(0, width):
                pos = topLeft
                pos += w * pixelRight
                pos += h * pixelBottom

                d = (pos - scene['camera']['center']).unit()
                pixel = rayTracer(scene, scene['camera']['center'], d, 4)
                f.write(f"{pixel.x*255} {pixel.y*255} {pixel.z*255}\n")
                # if(w==1):
                #     break
            # if(h==0):
            #     break
    end_time = time.time() - start_time
    print('Renderizado em ' + str(end_time) + ' segundos')
    f.close()

def parse(file):
    scene = {}
    with open(file) as f:
    # 1) Descrição de Câmera
        camera = {}
        aux = list(map(float,f.readline().split()))
        camera['center'] = vec3(aux[0],aux[1], aux[2])
        aux = list(map(float,f.readline().split()))
        camera['eye'] = vec3(aux[0],aux[1], aux[2])
        aux = list(map(float,f.readline().split()))
        up = (vec3(aux[0],aux[1], aux[2])).unit()
        aux_ = (camera['eye'] - camera['center']).unit()
        factor = aux_*(up.dot(aux_))/(aux_.squared_length())
        camera['up'] = (up - factor).unit()
        
        camera['fov'] = float(f.readline())
        
        scene['camera'] = camera

    # 2) Descrição de Luzes
        numLights = int(f.readline())
        lights = []
        for l in range (numLights):
            aux = list(map(float,f.readline().split()))
            light = {}
            light['pos'] = vec3(aux[0], aux[1], aux[2])
            light['rgb'] = vec3(aux[3], aux[4], aux[5])
            light['at1'] = aux[6]
            light['at2'] = aux[7]
            light['at3'] = aux[8]
            lights.append(light)  

        scene['lights'] = lights

    # 3) Descrição de Pigmentos
        numPigments = int(f.readline())
        pigments = []
        for l in range (numPigments):
            aux = f.readline().split()
            pigment = {}
            pigment['type'] = aux[0]
            if aux[0]=='solid':
                pigment['rgb'] = vec3(float(aux[1]), float(aux[2]), float(aux[3]))
            if aux[0]=='checker':
                pigment['rgb0'] = vec3(float(aux[1]), float(aux[2]), float(aux[3]))
                pigment['rgb1'] = vec3(float(aux[4]), float(aux[5]), float(aux[6]))
                pigment['size'] = float(aux[7])
            if aux[0]=='texmap':
                print("to do")
                f.readline()
                f.readline()
                pigment['type'] = 'solid'
                pigment['rgb'] = vec3(1,1,1)

            pigments.append(pigment)    

        scene['pigments'] = pigments

    # 4) Descrição de Acabamentos
        numFinish = int(f.readline())
        finishes = []
        for l in range (numFinish):
            aux = list(map(float,f.readline().split()))
            finish = {}
            finish['ka'] = aux[0]
            finish['kd'] = aux[1]
            finish['ks'] = aux[2]
            finish['a'] = aux[3]
            finish['kr'] = aux[4]
            finish['kt'] = aux[5]
            finish['ior'] = aux[6]
            finishes.append(finish)  

        scene['finishes'] = finishes

    # 5) Descrição de Objetos
        numObjects = int(f.readline())
        objects = []
        for l in range (numObjects):
            aux = f.readline().split()
            obj = {}
            obj['pigment'] = int(aux[0])
            obj['finish'] = int(aux[1])
            obj['type'] = aux[2]
            if aux[2]=='sphere':
                obj['center'] = vec3(float(aux[3]), float(aux[4]), float(aux[5]))
                obj['radius'] = float(aux[6])
            if aux[2]=='polyhedron':
                planes = []
                for w in range(int(aux[3])):
                    plane = {}
                    p = f.readline().split()
                    plane['a'] = float(p[0])
                    plane['b'] = float(p[1])
                    plane['c'] = float(p[2])
                    plane['d'] = float(p[3])
                    planes.append(plane)
                obj['planes'] = planes
            objects.append(obj)    

        scene['objects'] = objects
    
    return scene

def main():

    width = 800
    height = 600

    # width = 200
    # height = 150

    if(len(sys.argv) < 3):
        print("Forneça o nome dos arquivos de entrada e saída")
        sys.exit()
    else:
        inputFile  = sys.argv[1]
        outputFile = sys.argv[2]

    if(len(sys.argv) >= 5):
        width  = int(sys.argv[3])
        height = int(sys.argv[4])

    scene = parse(inputFile)
    print(scene)
    print("")

    computeScene(height, width, scene, outputFile)


if __name__ == '__main__':
    main()