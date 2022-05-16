
import cv2
from torch.autograd import Variable
import torch
import numpy as np
import open3d as o3d
import skimage.measure
from scipy import ndimage

def transform3x3(pc, T):
    # T: [4,4]
    # pc: [n, 3]
    # return: [n, 3]
    return (np.matmul(T[:3, :3], pc.T)).T
def transform4x4(pc, T):
    # T: [4,4]
    # pc: [n, 3]
    # return: [n, 3]
    return (np.matmul(T[:3, :3], pc.T) + T[:3, 3:4]).T
def v(var, cuda=True, volatile=False):
    if type(var) == torch.Tensor or type(var) == torch.DoubleTensor:
        res = Variable(var.float(), volatile=volatile)
    elif type(var) == np.ndarray:
        res = Variable(torch.from_numpy(var).float(), volatile=volatile)
    if cuda:
        res = res.cuda()
    return res

def npy(var):
    return var.data.cpu().numpy()

def convert_sdf_samples_to_trianglemesh(
    sdf,
    voxel_grid_origin,
    voxel_size,
):


    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        sdf, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # try writing to the ply file
    return mesh_points, faces

def write_mesh(fn, vert, face):
    mesh=o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vert)
    mesh.triangles = o3d.utility.Vector3iVector(face)
    o3d.io.write_triangle_mesh(fn, mesh)

def write_ply(fn, point, normal=None, color=None):

    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(point)
    if color is not None:
        if len(color.shape) == 1 and len(color)==3:
            color = np.tile(color[None,:],[len(point),1])
        ply.colors = o3d.utility.Vector3dVector(color)
    if normal is not None:
        ply.normals = o3d.utility.Vector3dVector(normal)
    o3d.io.write_point_cloud(fn, ply)

def pcloud_line(prev, cur, color=None):
    alpha = np.linspace(0, 1, 100)
    pcloud = prev[None, :] + alpha[:, None] * (cur - prev)[None, :]
    if color is None:
        pcolor = np.tile(np.array([0, 1, 0])[None, :], [pcloud.shape[0], 1])
    else:
        assert (len(color) == 3)
        pcolor = np.tile(np.array(color)[None, :], [pcloud.shape[0], 1])
    return pcloud, pcolor

def vis_rays(rays,depth):
    pcs = []
    for i in range(rays.shape[0]):
        #pc, _ = pcloud_line(rays[i, :3], rays[i,:3] + rays[i,3:6]*depth[i])
        if depth[i]!=-1:
            pc = rays[i,:3] + rays[i,3:6]*depth[i]
            pc = pc[None,:]
            pcs.append(pc)
    pcs = np.concatenate(pcs)
    write_ply('test.ply', pcs) 


def directional_rays():
    dim = 128
    tp = np.linspace(-1, 1, dim)
    ys, zs = np.meshgrid(tp, tp)
    ys = ys.flatten()
    zs = zs.flatten()
    xs = np.ones_like(ys) * 3
    pos = np.stack((xs, ys, zs), -1)
    direction = np.tile(np.array([-1, 0, 0])[None, :], [len(xs), 1])
    ray = np.concatenate((pos, direction), -1)
    return ray


def random_cam_rays(phi=None,theta=None,only_ray=True):
    K = np.array([140, 0, 64, 0, 140, 64, 0, 0, 1]).reshape(3, 3)
    r = 3.0
    if phi is None:
        phi = np.random.uniform(0, 2*np.pi)
    if theta is None:
        theta = np.random.uniform(0, 2*np.pi)
    pos = np.array([r*np.sin(phi), r*np.cos(phi)*np.cos(theta), r*np.cos(phi)*np.sin(theta)])
    lookat = -pos/np.linalg.norm(pos)
    up = np.array([0,1,0])
    right = np.cross(lookat, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, lookat)
    R = np.eye(4)
    R[:3, :] = np.stack((right, -up, lookat, pos), 1)

    h, w = 128, 128
    grid = np.stack(np.meshgrid(range(h), range(w)), -1)
    grid = grid.reshape(-1, 2)
    grid = np.concatenate((grid, np.ones([len(grid),1])), -1)

    pixel3d = np.matmul(np.linalg.inv(K), grid.T).T
    pixel3d = (np.matmul(R[:3,:3], pixel3d.T) + R[:3,3:4])
    ray = (pixel3d - R[:3,3:4]).T
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    
    ray_origin = np.tile(R[:3,3:4], [1, len(ray)]).T
    ray = np.concatenate((ray_origin, ray), -1)

    # visualize rays
    if 0:
        vis_rays(ray)

    if only_ray:
        return ray
    else:
        return ray, R, K

def vis_depth(depth, fn='test.png', ret=False):
    vis = depth.data.cpu().numpy()
    vis[vis== -1] = 0
    #vis[vis== 3] = 0
    #vis[vis== 4] = 0
    #vis[vis== 2] = 0
    vis = (vis) / (vis.max()+1e-16) * 255
    if ret:
        return vis
    else:
        cv2.imwrite(fn, vis)

def vis_pc(depth, rays,fn='test.ply', return_pc=False):
    depth_np = depth.data.cpu().numpy()
    rays_np = rays.data.cpu().numpy()
    pts = rays_np[:,:3] + depth_np.reshape(-1,1) * rays_np[:,3:6]
    
    mask = depth_np.flatten() != -1
    if return_pc:
        return pts[mask]
    else:
        write_ply(fn, 
                pts[mask])

def vis_pc_batch(depth, rays, fn, rgb=None):
    n = depth.shape[0]
    pc=[]
    pc_c=[]
    for i in range(n):
        pc.append(vis_pc(depth[i].view(128, 128), rays[i], return_pc=True))
        if rgb is not None:
            mask=npy(depth[i])!=-1
            pc_c.append(npy(rgb[i].reshape(-1,3)[mask]))

    pc = np.concatenate(pc)
    if rgb is not None:
        pc_c = np.concatenate(pc_c)
        write_ply(fn,pc,color=pc_c[:,::-1])
    else:
        write_ply(fn,pc)

def create_unit_sphere(radius=1.0, grid_w=1):
    dim = 64
    xs = np.linspace(-grid_w, grid_w, dim)
    grid = np.stack(
            np.meshgrid(xs, xs, xs, indexing='ij'), -1)
    grid = np.reshape(grid, [-1, 3])
    sdf = np.linalg.norm(grid, axis=-1) - radius
    sdf = sdf.reshape(dim, dim, dim)
    return sdf, np.array([-grid_w, -grid_w, -grid_w]), np.array([2*grid_w, 2*grid_w, 2*grid_w])

def create_texture(dim, offset=0):
    tp = np.linspace(-1,1,dim)
    xs, ys, zs = np.meshgrid(tp,tp,tp,indexing='ij')
    r = np.sin((xs-offset)/0.2)
    g = np.sin((ys-offset)/0.4)
    b = np.sin((zs-offset)/0.6)
    grid = np.stack((r,g,b),-1)
    feature = torch.from_numpy(grid).unsqueeze(0).float().cuda()
    feature = (feature+1)/2
    return feature


def vis_texture(x, fn='test.png', ret=False):
    rgb = x.data.cpu().numpy()*255
    if ret:
        return rgb
    else:
        cv2.imwrite(fn,rgb)

def compare_texture(x, y, fn='test.png'):
    n = x.shape[0]
    vis = []
    for i in range(n):
        tp0=vis_texture(x[i], ret=True)
        tp1=vis_texture(y[i], ret=True)
        vis.append(np.concatenate((tp0,tp1),1))
    vis = np.concatenate(vis)
    cv2.imwrite(fn, vis)
    


def compare_img(x, y, fn='test.png'):
    n = x.shape[0]
    vis = []
    for i in range(n):
        tp0=npy(x[i])
        tp1=npy(y[i])
        vis.append(np.concatenate((tp0,tp1),1))
    vis = np.concatenate(vis)
    cv2.imwrite(fn, vis)

def voxel_carving(grid, origin, poses, K, masks):
    grid_freespace=np.zeros([len(grid)])
    n = len(poses)
    print('begin voxel carving ..')
    for j in range(n):
        world2cam=np.linalg.inv(poses[j])
        mask = masks[j].astype('bool').reshape(128,128)
        edt = ndimage.distance_transform_edt((~mask).astype('float'))
        mask = edt<4.0
        grid_trans = transform4x4(grid,world2cam)
        grid_trans = transform3x3(grid_trans,K)
        grid_trans = grid_trans[:,:2]/grid_trans[:,2:3]
        mask0=(grid_trans[:,0]>=0)&(grid_trans[:,0]<=128-1)
        mask1=(grid_trans[:,1]>=0)&(grid_trans[:,1]<=128-1)
        grid_trans_clip=grid_trans.clip(0,128-1).astype('int')
        mask2= mask[grid_trans_clip[:,1],grid_trans_clip[:,0]]
        mask_non_freespace=mask0&mask1&mask2
        grid_freespace[~mask_non_freespace.astype('bool')]+=1

    print('finish voxel carving ..')
    xyz_pos = grid[grid_freespace!=0]
    xyz_neg = grid[grid_freespace==0]

    grid_freespace = grid_freespace.reshape(64,64,64)
    if 0:
        tp = np.stack(np.where(grid_freespace==0))
        x =-0.5+tp[0,:]*(1/63.0)
        y =-0.5+tp[1,:]*(1/63.0)
        z =-0.5+tp[2,:]*(1/63.0)
        write_ply('test1.ply',np.stack((x,y,z),-1))

    voxel_size = 2 * -origin[0] / (grid_freespace.shape[0]-1) 
    sdf = (grid_freespace  > 0)
    sdf_pos = ndimage.distance_transform_edt(sdf)
    sdf_neg = ndimage.distance_transform_edt(~sdf)
    sdf = sdf_pos * sdf + -sdf_neg * (1-sdf) 
    sdf = sdf * voxel_size

    mask_pos = (grid_freespace.flatten()>0)
    mask_neg = ~mask_pos
    return sdf, mask_pos, mask_neg

