import cv2
import numpy as np
import mediapipe as mp
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import threading
import queue
import time
import pickle
import os

from tracker_base import Tracker

from mp_2_flame import MP_2_FLAME

## Enviroment Setup
import os, sys
DIR = os.getcwd()
WORKING_DIR = 'E:\ICT-FaceKit-master\ICT-FaceKit-master\FaceXModel'
os.chdir(WORKING_DIR) # change the working directory to the project's absolute path
print("Current Working Directory: ", os.getcwd())

sys.path.append(WORKING_DIR)
sys.path.append('./utils/flame_lib/')
sys.path.append('./utils/flame_fitting/')
sys.path.append('./utils/face_parsing/')
sys.path.append('./utils/decalib/')
sys.path.append('./utils/mesh_renderer')
sys.path.append('./utils/scene')

mp2flame = MP_2_FLAME(mappings_path='c:\\Users\\Mi Manchi\\Desktop\\FACE_3D_NEW\\Mappings')

# ========== 全局变量 ==========
model_path = 'face_landmarker_v2_with_blendshapes.task'  # 需提前下载模型
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# FLAME 模型参数
flame_model = None
flame_template_vertices = None
flame_faces = None
flame_tex_params = None

frame_count = 0  # 用于跟踪帧数（例如保存模型时使用）

# 表情PCA相关
expression_basis = None  # 表情PCA基 (100维)
avg_vertices = None
faces = None
normals = None
face_normals = None

vbo = None
normals_vbo = None
ibo = None

update_queue = queue.Queue()
window_width, window_height = 800, 600
angle_x, angle_y = 0, 0  # 简单的相机旋转

# 头部姿态矩阵
head_pose_matrix = np.eye(4, dtype=np.float32)

# 用于头部姿态的3D模型点（关键面部点）
model_points = np.array([
    [0.0, 0.0, 0.0],             # 鼻尖
    [0.0, -330.0, -65.0],        # 下巴
    [-225.0, 170.0, -135.0],     # 左眼左角
    [225.0, 170.0, -135.0],      # 右眼右角
    [-150.0, -150.0, -125.0],    # 左嘴角
    [150.0, -150.0, -125.0]      # 右嘴角
], dtype=np.float32)

## Computing Device
device = 'cuda:0'
import torch
torch.cuda.set_device(device) # this will solve the problem that OpenGL not on the same device with torch tensors

tracker_cfg = {
    'mediapipe_face_landmarker_v2_path': 'C:\\Users\\Mi Manchi\\Desktop\\FACE_3D_NEW\\models\\face_landmarker_v2_with_blendshapes.task',
    'flame_model_path': 'c:\\Users\\Mi Manchi\\Desktop\\FACE_3D_NEW\\models\\FLAME2020\\generic_model.pkl',
    'flame_lmk_embedding_path': 'C:\\Users\\Mi Manchi\\Desktop\\FACE_3D_NEW\\models\\landmark_embedding.npy',
    'tex_space_path': 'C:\\Users\\Mi Manchi\\Desktop\\FACE_3D_NEW\\models\\FLAME_albedo_from_BFM.npz',
    'face_parsing_model_path': 'c:\\Users\\Mi Manchi\\Desktop\\FACE_3D_NEW\\models\\79999_iter.pth',
    'template_mesh_file_path': 'C:\\Users\\Mi Manchi\\Desktop\\FACE_3D_NEW\\models\\head_template.obj',
    'result_img_size': 512,
    'device': device,
}

tracker = Tracker(tracker_cfg)

# Initialize blendshape and other arrays
shape = np.zeros([1, 300], dtype=np.float32)
exp = np.zeros([1, 100], dtype=np.float32)
pose = np.zeros([1, 6], dtype=np.float32)
eye_pose = np.zeros([1, 6], dtype=np.float32)

def convert_landmarks_to_expression(landmarks):
    """
    将面部特征点转换为FLAME表情参数
    :param landmarks: 68个面部特征点的坐标
    :return: 100维的表情参数数组
    """
    # 使用mp_2_flame模块将MediaPipe特征点转换为FLAME参数
    try:
        # 调用MP_2_FLAME模块进行转换
        expression_params = mp2flame.convert(landmarks)
        return expression_params
    except Exception as e:
        print(f"Error converting landmarks to expression: {e}")
        return np.zeros(100, dtype=np.float32)

# ========== 计算法线 (优化版) ==========
def compute_normals_vectorized_optimized(vertices, faces, face_normals=None):
    if face_normals is None:
        v0 = vertices[faces[:,0]]
        v1 = vertices[faces[:,1]]
        v2 = vertices[faces[:,2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        norms_len = np.linalg.norm(face_normals, axis=1, keepdims=True)
        norms_len[norms_len == 0] = 1
        face_normals /= norms_len
    
    normals = np.zeros(vertices.shape, dtype=np.float32)
    np.add.at(normals, faces[:,0], face_normals)
    np.add.at(normals, faces[:,1], face_normals)
    np.add.at(normals, faces[:,2], face_normals)
    
    norms_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norms_len[norms_len == 0] = 1
    normals /= norms_len
    return normals, face_normals

# ========== 归一化顶点 ==========
def normalize_vertices(vertices):
    min_corner = np.min(vertices, axis=0)
    max_corner = np.max(vertices, axis=0)
    center = (min_corner + max_corner) / 2
    size = np.linalg.norm(max_corner - min_corner)
    if size == 0:
        size = 1
    return (vertices - center) / size * 2

# ========== 加载FLAME模型 ==========
def load_flame_model(pkl_path, obj_path):
    """加载FLAME模型参数和模板网格"""
    global flame_model, flame_template_vertices, flame_faces
    global expression_basis, avg_vertices, faces
    
    # 加载FLAME模型参数
    pkl_path = os.path.join('c:\\Users\\Mi Manchi\\Desktop\\FACE_3D_NEW\\models\\FLAME2020', 'generic_model.pkl')
    with open(pkl_path, 'rb') as f:
        flame_model = pickle.load(f, encoding='latin1')
    
    # 获取表情PCA基 (后100维)
    expression_basis = flame_model['shapedirs'][..., -100:]
    
    # 加载模板网格
    vertices, faces = load_face_model('c:\\Users\\Mi Manchi\\Desktop\\FACE_3D_NEW\\models\\head_template.obj', normalize=False)
    flame_template_vertices = vertices
    flame_faces = faces
    
    # 初始化参数
    avg_vertices = vertices.copy()
    faces = flame_faces.copy()

def save_model(vertices, faces, filename):
    """将模型保存为OBJ文件"""
    with open(filename, 'w') as f:
        # 写入顶点
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        # 写入面
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"Model saved to {filename}")

# ========== 应用FLAME表情参数 ==========
def apply_flame_expression(vertices, exp, pose, eye_pose):
    """应用FLAME表情参数到模板网格"""
    global frame_count, shape
    """应用FLAME表情参数到模板网格"""
    if exp is None or len(exp) != 100:
        return vertices
    
    frame_count += 1  # 增加帧计数
    
    # 确保exp_params形状正确 (100,)
#    exp = np.asarray(exp).reshape(-1)

    # 打印前10个表情参数
#    print("前10个表情参数:", " ".join([f"{x:.4f}" for x in exp_params[:10]]))
    
    # 将表情参数应用到表情PCA基
    # expression_basis形状应为 (100, n_vertices*3)
    # 我们需要计算 (exp_params @ expression_basis).reshape(n_vertices, 3)
    
    # 先将expression_basis从(5023,3,100)转换为(100,5023*3)
#    basis_reshaped = np.transpose(expression_basis, (2, 0, 1)).reshape(100, -1)
    
    # 计算变形: (100,) @ (100, 5023*3) -> (5023*3,)
#    exp_deformation = np.dot(exp, basis_reshaped).reshape(-1, 3)
    
    # Prepare Tensors
    shapeT = torch.from_numpy(shape).to(device)
    expT = torch.from_numpy(exp).to(device)
    poseT = torch.from_numpy(pose).to(device)
    eye_poseT = torch.from_numpy(eye_pose).to(device)

    expT = expT.unsqueeze(0)
    eye_poseT = eye_poseT.unsqueeze(0)

    # 应用变形
    # FLAME Reconstruction
    vertices, _, _ = tracker.flame(shape_params=shapeT, expression_params=expT, pose_params=poseT, eye_pose_params=eye_poseT)

    vertices = vertices.cpu().numpy()  # GPU → CPU → NumPy
    vertices = vertices.squeeze(0)  # 从 (1, 5023, 3) → (5023, 3)
#    deformed_vertices = vertices + exp_deformation
    
    return vertices

# ========== VBO初始化 ==========
def init_vbos():
    global vbo, normals_vbo, ibo, faces
    
    vbo = glGenBuffers(1)
    normals_vbo = glGenBuffers(1)
    ibo = glGenBuffers(1)
    
    # 初始化顶点数据
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, avg_vertices.nbytes, avg_vertices, GL_DYNAMIC_DRAW)
    
    # 初始化法线数据
    glBindBuffer(GL_ARRAY_BUFFER, normals_vbo)
    glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_DYNAMIC_DRAW)
    
    # 初始化索引数据
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)
    
    # 解绑
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

def update_vbos(vertices, normals):
    global vbo, normals_vbo
    
    # 确保数组是连续的
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    normals = np.ascontiguousarray(normals, dtype=np.float32)
    
    # 更新顶点VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
    
    # 更新法线VBO
    glBindBuffer(GL_ARRAY_BUFFER, normals_vbo)
    glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_DYNAMIC_DRAW)
    
    # 解绑
    glBindBuffer(GL_ARRAY_BUFFER, 0)

# ========== MediaPipe初始化 ==========
def setup_mediapipe():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        num_faces=1
    )
    options.base_options.model_asset_path = 'c:\\Users\\Mi Manchi\\Desktop\\FACE_3D_NEW\\models\\face_landmarker_v2_with_blendshapes.task'
    return FaceLandmarker.create_from_options(options)

# ========== 头部姿态估计 ==========
def estimate_head_pose(image, landmarks):
    if len(landmarks) < 446:  # 确保有足够的点
        return None
    
    # 选择关键点
    keypoints = [1, 152, 226, 446, 57, 287]

    # 只提取前两个分量 (x, y) - 忽略z坐标
    image_points = np.array([
        [landmarks[i][0], landmarks[i][1]] for i in keypoints
    ], dtype=np.float32)
    
    # 转换为像素坐标
    h, w = image.shape[:2]
    image_points[:, 0] *= w
    image_points[:, 1] *= h
    
    # 相机内参（近似值）
    focal_length = w
    center = (w/2, h/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 畸变系数（假设为0）
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
 
    # 使用solvePnP计算姿态
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, 
        image_points, 
        camera_matrix, 
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return None
    
    # 将旋转向量转换为旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # 单位转换（毫米→米）
    translation_vector *= 0.001
    
    # 创建4x4变换矩阵
    pose_matrix = np.eye(4, dtype=np.float32)
    pose_matrix[:3, :3] = rotation_matrix
    pose_matrix[:3, 3] = 0  # 强制平移分量为0
    
    # 转换坐标系：OpenCV到OpenGL
    cv_to_gl = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ], dtype=np.float32)
    
    return cv_to_gl @ pose_matrix

def load_face_model(obj_path, normalize=False):
    obj_path = os.path.abspath(obj_path)  # 确保使用绝对路径
    print(f"Loading face model from: {obj_path}")
    try:
        vertices = []
        faces = []
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()[1:4]
                    vertices.append([float(p) for p in parts])
                elif line.startswith('f '):
                    tokens = line.strip().split()[1:]
                    face_indices = []
                    for t in tokens:
                        idx = t.split('/')[0]
                        face_indices.append(int(idx) - 1)
                    if len(face_indices) == 3:
                        faces.append(face_indices)
                    elif len(face_indices) == 4:
                        faces.append([face_indices[0], face_indices[1], face_indices[2]])
                        faces.append([face_indices[0], face_indices[2], face_indices[3]])
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        if normalize:
            vertices = normalize_vertices(vertices)
        return vertices, faces
    except Exception as e:
        print(f"Error loading face model: {e}")
        raise

# ========== 处理帧 ==========
def process_frame(frame, frame_timestamp, face_landmarker):
    global head_pose_matrix, exp, pose, eye_pose
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = face_landmarker.detect_for_video(mp_image, frame_timestamp)
    weights = None
    
    if result.face_landmarks and len(result.face_landmarks) > 0:
        # 获取BlendShape权重
        if result.face_blendshapes:
            blendshapes = result.face_blendshapes[0]
            weights = np.array([bs.score for bs in blendshapes], dtype=np.float32)
        
        # 估计头部姿态
        landmarks = [(lm.x, lm.y, lm.z) for lm in result.face_landmarks[0]]
        pose_matrix = estimate_head_pose(frame, landmarks)
        if pose_matrix is not None:
            head_pose_matrix = pose_matrix
        
        # 绘制关键点
        for landmark in result.face_landmarks[0]:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # 计算FLAME参数
    exp, pose, eye_pose = mp2flame.convert(blendshape_scores=weights)
    
    return frame, exp, pose, eye_pose

# ========== 摄像头线程 ==========
def camera_thread(face_landmarker):
    global exp, pose, eye_pose, update_queue
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    frame_timestamp = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        frame = cv2.flip(frame, 1)  # 水平翻转
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame, exp, pose, eye_pose = process_frame(frame_rgb, frame_timestamp, face_landmarker)
        
        if exp is not None:
            update_queue.put(exp.copy())
            
        cv2.imshow("Face Tracking", cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
        frame_timestamp += int(1000 / cap.get(cv2.CAP_PROP_FPS))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# ========== OpenGL 渲染函数 ==========
def display():
    global exp, pose, eye_pose, avg_vertices, frame_count, faces, normals, face_normals
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    
    # 设置相机
    gluLookAt(0, 0, 3, 0, 0, 0, 0, 1, 0)
    glRotatef(angle_x, 1, 0, 0)
    glRotatef(angle_y, 0, 1, 0)
    glTranslatef(0, 0, 0.6)
    
    # 应用头部姿态
    glMultMatrixf(head_pose_matrix.T)
    
    # 应用FLAME表情参数
    if exp is not None:
        deformed = apply_flame_expression(flame_template_vertices, exp, pose, eye_pose)
        deformed = normalize_vertices(deformed)
#        print(deformed.shape, faces.shape, face_normals.shape)
        normals, face_normals = compute_normals_vectorized_optimized(deformed, faces, face_normals)
    else:
        deformed = normalize_vertices(flame_template_vertices)
        normals, face_normals = compute_normals_vectorized_optimized(deformed, faces, face_normals)

    # 在第100帧保存纯表情模型（不包含姿态变化）
    if frame_count == 100:
        # 创建保存目录（如果不存在）
        os.makedirs("saved_models", exist_ok=True)
        # 保存模型
        save_path = os.path.join("saved_models", "expression_only_frame_100.obj")
        save_model(deformed, flame_faces, save_path)
        print(f"Saved expression-only model at frame {frame_count} to {save_path}")
    
    # 更新VBO数据
    update_vbos(deformed, normals)
    
    # 启用光照
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    light_pos = [1.0, 1.0, 1.0, 0.0]
    glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
    
    # 设置材质属性
    glColor3f(0.7, 0.7, 0.7)
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.7, 0.7, 0.7, 1.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.3, 0.3, 0.3, 1.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 50.0)
    
    # 使用VBO渲染
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glVertexPointer(3, GL_FLOAT, 0, None)
    
    glBindBuffer(GL_ARRAY_BUFFER, normals_vbo)
    glNormalPointer(GL_FLOAT, 0, None)
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
    glDrawElements(GL_TRIANGLES, len(faces)*3, GL_UNSIGNED_INT, None)
    
    # 清理状态
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_NORMAL_ARRAY)
    glDisable(GL_LIGHTING)
    
    glutSwapBuffers()

def idle():
    global exp
    try:
        while True:
            exp = update_queue.get_nowait()
    except queue.Empty:
        pass
    glutPostRedisplay()

def reshape(width, height):
    global window_width, window_height
    window_width, window_height = width, height
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, width / height, 0.1, 100)
    glMatrixMode(GL_MODELVIEW)

def check_gl_error():
    error = glGetError()
    if error != GL_NO_ERROR:
        print(f"OpenGL error: {error}")

# ========== 主程序 ==========
def main():
    global avg_vertices, faces, normals, face_normals
    
    print("Loading FLAME model...")
    load_flame_model("generic_model.pkl", "head_template.obj")

    print("Computing initial normals...")
    normals, face_normals = compute_normals_vectorized_optimized(avg_vertices, faces)

    print("Setting up MediaPipe...")
    face_landmarker = setup_mediapipe()

    print("Initializing OpenGL...")
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(window_width, window_height)
    glutCreateWindow(b"FLAME Face Model with Head Pose Tracking")
    
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_COLOR_MATERIAL)
    glClearColor(0.1, 0.1, 0.1, 1.0)
    
    init_vbos()  # 初始化VBO
    
    print("Starting camera thread...")
    threading.Thread(target=camera_thread, args=(face_landmarker,), daemon=True).start()

    glutDisplayFunc(display)
    glutIdleFunc(idle)
    glutReshapeFunc(reshape)
    glutMainLoop()

if __name__ == "__main__":
    main()
