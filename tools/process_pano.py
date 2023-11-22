import cv2
import numpy as np
import matplotlib.pyplot as plt

class Perspective:
    def __init__(self, img_name):
        self._img = img_name
        
        [self._height, self._width, _] = self._img.shape
        print(self._img.shape)
    

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        equ_h = self._height
        equ_w = self._width
        
        # max x,y (จาก perspective ครึ่งรูป)
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0
        
        # คิด FOV
        wFOV = FOV
        hFOV = float(height) / width * wFOV

        w_len = np.tan(np.radians(wFOV / 2.0))
        h_len = np.tan(np.radians(hFOV / 2.0))

        # สร้าง xyz จาก FOV ที่ต้องการ
        x_map = np.ones([height, width], np.float32)
        y_map = np.tile(np.linspace(-w_len, w_len,width), [height,1])
        z_map = -np.tile(np.linspace(-h_len, h_len,height), [width,1]).T # transpose
        
        # หาค่า r
        D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
        
        xyz = np.stack((x_map,y_map,z_map),axis=2)/np.repeat(D[:, :, np.newaxis], 3, axis=2)
        
        # rotation matrix
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA)) # rotation around y-axis (use theta) 
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI)) # rotation z-axis (use phi)
        xyz = xyz.reshape([height * width, 3]).T # transpose of xyz
        # use rotation matrix with picture (cartesian coordinates - xyz)
        xyz = np.dot(R1, xyz)  
        xyz = np.dot(R2, xyz).T
        
        # uv mapping
        lat = np.arcsin(xyz[:, 2]) # phi (v)
        lon = np.arctan2(xyz[:, 1] , xyz[:, 0]) # theta (u)
        lon = lon.reshape([height, width]) / np.pi * 180 # เปลี่ยนเป็น degree แล้ว
        lat = -lat.reshape([height, width]) / np.pi * 180 # เปลี่ยนเป็น degree แล้ว
        lon = ((lon / 180) * equ_cx + 0.5) + equ_cx -0.5 # /180 เพราะเป็นพิกัดแนว y (-180,180) # opencv 0.5 shift
        lat = ((lat / 90)  * equ_cy + 0.5) + equ_cy -0.5# /90 เพราะเป็นพิกัดแนว x (-90,90) # opencv 0.5 shift

        # mapping image to desired coordinates 
        persp = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        return persp
    
class Equirectangular:
    def __init__(self, img_name , FOV, THETA, PHI ,type_img='img'):
        self._img = img_name
        if(type_img == 'mask'):
            self._img = cv2.bitwise_not(self._img)
        [self._height, self._width, _] = self._img.shape
        self.wFOV = FOV
        self.THETA = THETA
        self.PHI = PHI
#         self.hFOV = float(self._height) / self._width * FOV
        self.hFOV = np.rad2deg(2 * np.arctan(self._height * np.tan(np.radians(FOV/2)) / self._width))
        self.w_len = np.tan(np.radians(self.wFOV / 2.0))
        self.h_len = np.tan(np.radians(self.hFOV / 2.0))

    

    def GetEquirec(self,height,width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #
        
        # meshgrid (สร้าง grid ที่มีจุดกระจายอยู่เป็นคู่ตน. x,y ทุกจุด) ทั้งรูปภาพ y(height) -> (-90,90), x(width) -> (-180,180) 
        x,y = np.meshgrid(np.linspace(-180, 180,width),np.linspace(90,-90,height))
        
        # 
        x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
        y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
        z_map = np.sin(np.radians(y))

        xyz = np.stack((x_map,y_map,z_map),axis=2)
        
        # rotation matrix
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(self.THETA)) # rotation around z-axis
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-self.PHI)) # rotation around y-axis
        
        # inverse matrix
        R1 = np.linalg.inv(R1)
        R2 = np.linalg.inv(R2)
        
        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R2, xyz)
        xyz = np.dot(R1, xyz).T

        xyz = xyz.reshape([height , width, 3])
        inverse_mask = np.where(xyz[:,:,0]>0,1,0) # จำกัดเขตที่จะสร้างภาพ equirectangular 

        xyz[:,:] = xyz[:,:]/np.repeat(xyz[:,:,0][:, :, np.newaxis], 3, axis=2)
        
        # w_len, h_len -> field of view
        # หาตน. ที่มี xyz ตรงกันระหว่าง coordinates เป็นผืนผ้า equirec กับ 
        # lat lon - พิกัดบนล่าง
        lon_map = np.where((-self.w_len<xyz[:,:,1])&(xyz[:,:,1]<self.w_len)&(-self.h_len<xyz[:,:,2])
                    &(xyz[:,:,2]<self.h_len),(xyz[:,:,1]+self.w_len)/2/self.w_len*self._width,0)
        lat_map = np.where((-self.w_len<xyz[:,:,1])&(xyz[:,:,1]<self.w_len)&(-self.h_len<xyz[:,:,2])
                    &(xyz[:,:,2]<self.h_len),(-xyz[:,:,2]+self.h_len)/2/self.h_len*self._height,0)
        mask = np.where((-self.w_len<xyz[:,:,1])&(xyz[:,:,1]<self.w_len)&(-self.h_len<xyz[:,:,2])
                    &(xyz[:,:,2]<self.h_len),1,0)
        
        # map pixel ตาม lat lon coordinates
        persp = cv2.remap(self._img, lon_map.astype(np.float32), lat_map.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        
        # เอาเฉพาะช่วงล่าง
        mask = mask * inverse_mask
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        persp = persp * mask
        
#         # ค่อยมาเปลี่ยนอีกที
#         for j in range(height):
#             for i in range(width):
#         #       # Replace the pixels in the original image with the replacement image's pixels
#                 self._oimg[j,i] = persp[j, i]
        
        
        return persp , mask
    
def inpaint_pano(inpaint_pano_img,mask_pano_img,car_pano_img):
    inpaint_pano_img = inpaint_pano_img/255.0
    mask_pano_img = mask_pano_img/255.0
    car_pano_img = car_pano_img/255.0
    inpaint_fill_pano = ((mask_pano_img) * inpaint_pano_img) + ((1-mask_pano_img) * car_pano_img)
    return inpaint_fill_pano