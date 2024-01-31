# 3DDFA-V1 XLR8TD
#!/usr/bin/env python3
# coding: utf-8


"""
3DDFA-V1 - XLR8TD
"""

__author__ = 'cleardusk, VictorHManrique, EugeniaVirtualHumans'

# Kivy
from kivy.app import App
from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.properties import StringProperty, ListProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.config import Config
from kivy.uix.image import Image
from kivy.uix.image import AsyncImage
from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.core.text import LabelBase
from kivy.properties import ObjectProperty
from kivy.graphics.texture import Texture
from kivy.core.image import Image as CoreImage

# KivyMD
from kivymd.theming import ThemableBehavior
from kivymd.uix.list import MDList
from kivymd.uix.screen import MDScreen
from kivymd.uix.list import OneLineIconListItem, MDList
from kivymd.icon_definitions import md_icons
from kivymd.uix.button import MDRectangleFlatButton
from kivymd.uix.card import MDCard

# OpenCV
import cv2

# Model 
import torch
import torchvision.transforms as transforms

# Architectures
import mobilenet_v1
import efficientnetlite
# import mobile_former
import ghostnet

# More
import numpy as np
import csv
import queue
import cv2
import dlib
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose
from utils.render import get_depths_image, cget_depths_image, cpncc
from utils.paf import gen_img_paf
import argparse
import torch.backends.cudnn as cudnn
import sys
import threading
from threading import Event

STD_SIZE = 120

# Tamaño de la ventana
from kivy.core.window import Window
Window.size=(800, 500)

# Frame rate
import time

class XLR8TD(Image):

    def __init__(self, **kwargs):
        super(XLR8TD, self).__init__(**kwargs)
        # Model    
        parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
        parser.add_argument('-f', '--files', nargs='+',
                            help='image files paths fed into network, single or multiple images')
        parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
        parser.add_argument('--show_flg', default='true', type=str2bool, help='whether show the visualization result')
        parser.add_argument('--bbox_init', default='one', type=str,
                            help='one|two: one-step bbox initialization or two-step')
        parser.add_argument('--dump_res', default='true', type=str2bool, help='whether write out the visualization image')
        parser.add_argument('--dump_vertex', default='false', type=str2bool,
                            help='whether write out the dense face vertices to mat')
        parser.add_argument('--dump_ply', default='true', type=str2bool)
        parser.add_argument('--dump_pts', default='true', type=str2bool)
        parser.add_argument('--dump_roi_box', default='false', type=str2bool)
        parser.add_argument('--dump_pose', default='true', type=str2bool)
        parser.add_argument('--dump_depth', default='true', type=str2bool)
        parser.add_argument('--dump_pncc', default='true', type=str2bool)
        parser.add_argument('--dump_paf', default='false', type=str2bool)
        parser.add_argument('--paf_size', default=3, type=int, help='PAF feature kernel size')
        parser.add_argument('--dump_obj', default='true', type=str2bool)
        parser.add_argument('--dlib_bbox', default='true', type=str2bool, help='whether use dlib to predict bbox')
        parser.add_argument('--dlib_landmark', default='true', type=str2bool,
                            help='whether use dlib landmark to crop image')
        global args
        args = parser.parse_args()

        # 1. load pre-tained model
        
        # MobileNetV1
        checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
        arch = 'mobilenet_1'
        
        # EfficientNetLite
        # checkpoint_fp = 'models/phase1_pdc_checkpoint_efficientnetlite_epoch_34.pth.tar'
        #checkpoint_fp = 'models/phase1_wpdc_checkpoint_efficientnetlite_epoch_7.pth.tar'
        #arch = 'efficientnetlite'# GLOBAL
        
        # GhostNet
        # checkpoint_fp = 'models/phase1_pdc_checkpoint_ghostnet_epoch_14.pth.tar'
        # arch = 'ghostnet'
        
        # MobileFormer
        # checkpoint_fp = 'models/phase1_pdc_checkpoint_mobileformer_epoch_9.pth.tar'
        # arch = 'mobileformer'

        checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
        
        global model
        model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)
        # model = getattr(efficientnetlite, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)
        # model = getattr(ghostnet, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)
        # model = getattr(mobileformer, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

        global model_dict
        model_dict = model.state_dict()
        # because the model is trained by multiple gpus, prefix module should be removed
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]
        model.load_state_dict(model_dict)
        if args.mode == 'gpu':
            cudnn.benchmark = True
            model = model.cuda()
        model.eval()

        # 2. load dlib model for face detection and landmark used for face cropping
        if args.dlib_landmark:
            dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
            global face_regressor
            face_regressor = dlib.shape_predictor(dlib_landmark_model)
        if args.dlib_bbox:
            global face_detector
            face_detector = dlib.get_frontal_face_detector()

        # 3. forward
        global tri
        tri = sio.loadmat('visualize/tri.mat')['tri']
        global transform
        transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
        self.capture = None

    def start(self, capture):
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / 60.0)

    def update(self, dt):
    
        # Shared resource
        shared_resource = []

        # Condition variable
        condition = threading.Condition()   

        def own_face_detection():
            if args.dlib_bbox:
                rects = face_detector(frame, 1)
                data = q.put(rects)  
            else:
                rects = []
                data = q.put(rects)
                
        def own_prediction():
            # two-step for more accurate bbox to crop face
            if args.bbox_init == 'two':
                roi_box = parse_roi_box_from_landmark(pts68)
                img_step2 = crop_img(frame, roi_box)
                img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
                input = transform(img_step2).unsqueeze(0)
                with torch.no_grad():
                    if args.mode == 'gpu':
                        input = input.cuda()
                    param = model(input)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
                pts68 = predict_68pts(param, roi_box)
            
        def own_pose():
            pts_res.append(pts68)
            P, pose = parse_pose(param)
            Ps.append(P)
            poses.append(pose)
            shared_resource.append(poses)
            condition.notify()
        
        def own_dense_vertices():
            # pose_thread.join()
            with condition:
                if args.dump_ply or args.dump_vertex or args.dump_depth or args.dump_pncc or args.dump_obj:
                    vertices = predict_dense(param, roi_box)
                    vertices_lst.append(vertices)
                    shared_resource.append(vertices_lst)
                    condition.notify()
                
        def own_depth():
            # dense_vertices_thread.join()
            with condition:
                while not shared_resource:
                    condition.wait()
                vertices_lst = shared_resource.pop(0)
                depths_img = cget_depths_image(frame, vertices_lst, tri - 1)  # cython version
                data2 = q.put(depths_img)	
            
        global fps, frame, q, vertices, vertices_lst
        global pts68, roi_box, img_step2, input, param
        global pts_res, P, pose, Ps, poses
        q = queue.Queue()
        
        # THREADS
        face_detection_thread = threading.Thread(target=own_face_detection)
        prediction_thread = threading.Thread(target=own_prediction)
        pose_thread = threading.Thread(target=own_pose)
        dense_vertices_thread = threading.Thread(target=own_dense_vertices)
        depth_thread = threading.Thread(target=own_depth)
        
        # GLOBAL 
        start_time = time.time() # start time of the loop                                             
        
        # CAPTURE
        return_value, frame = self.capture.read()
        
        # REDUCE RESOLUTION
        frame = cv2.resize(frame, (320,240), interpolation = cv2.INTER_LINEAR)

        # FACE DETECTOR
        face_detection_thread.start()
        face_detection_thread.join()
        data = q.get()
        
        if len(data) == 0:
            texture = self.texture
            frame = cv2.resize(frame, (640,480))
            w, h = frame.shape[1], frame.shape[0]
            self.texture = texture = Texture.create(size=(w, h))
            texture.flip_vertical()
            texture.blit_buffer(bytes(frame), colorfmt='bgr')
            self.canvas.ask_update()
            
        else: 
            pts_res = []
            Ps = []  # Camera matrix collection
            poses = []  # pose collection, [todo: validate it]
            vertices_lst = []  # store multiple face vertices
            ind = 0
            
            # CROPPING
            # for rect in rects:
            for rect in data:
                # whether use dlib landmark to crop image, if not, use only face bbox to calc roi bbox for cropping
                if args.dlib_landmark:
                    # - use landmark for cropping
                    pts = face_regressor(frame, rect).parts()
                    pts = np.array([[pt.x, pt.y] for pt in pts]).T
                    roi_box = parse_roi_box_from_landmark(pts)
                else:
                    # - use detected face bbox
                    bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                    roi_box = parse_roi_box_from_bbox(bbox)

                img = crop_img(frame, roi_box)

                # forward: one step  
                img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
                input = transform(img).unsqueeze(0)
                with torch.no_grad():
                    if args.mode == 'gpu':
                        input = input.cuda()
                    param = model(input)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
                    
                # PREDICTION 68 PTS  
                pts68 = predict_68pts(param, roi_box)

                # PREDICTION
                prediction_thread.start()

                # POSE
                # pose_thread.start()

                # DENSE 3D VERTICES
                dense_vertices_thread.start()
                    
                # DEPTH
                depth_thread.start()
                depth_thread.join()
                    
            # CONVERTION
            model_image_converted = q.get()
            model_image_converted = cv2.resize(model_image_converted, (640,480))
            
            # RENDER
            texture = self.texture
            w, h = model_image_converted.shape[1], model_image_converted.shape[0]
            self.texture = texture = Texture.create(size=(w, h))
            texture.flip_vertical()
            # texture.blit_buffer(bytes(model_image_converted), colorfmt='luminance_alpha', bufferfmt='ushort')
            texture.blit_buffer(bytes(model_image_converted), colorfmt='bgra', bufferfmt='ushort')
            self.canvas.ask_update()
        
            # GLOBAL
            fps = 1.0 / (time.time() - start_time)
            print("FPS: ", fps)
            # with open('ui_mobilenetv1.csv', 'a') as f:
                # writer = csv.writer(f)
                # writer.writerow([fps])

capture = None

class Home(BoxLayout):

    def init_julen(self):
        pass
        
    def dostart_camera(self, *largs):
        global capture
        capture = cv2.VideoCapture(0)
        self.ids.camera_view.start(capture)

class xlr8tdApp(App):

    def build(self):
        Window.clearcolor = (0,0,0,1)
        homeWin = Home()
        homeWin.init_julen()
        return homeWin

    def on_stop(self):
        global capture
        if capture:
            capture.release()
            capture = None

xlr8tdApp().run()
