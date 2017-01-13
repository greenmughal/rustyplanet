// Copyright 2014 The Gfx-rs Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

extern crate cgmath;
#[macro_use]
extern crate gfx;
extern crate gfx_app;
extern crate rand;
extern crate genmesh;
extern crate sfml;
extern crate multi_input;
extern crate noise;
extern crate winit;
use rand::random;
use noise::{Seed, perlin2};
use rand::Rng;
use cgmath::{SquareMatrix, Matrix4, Point3, Vector3};
use cgmath::{Transform, AffineMatrix3};
pub use gfx::format::{DepthStencil};
pub use gfx_app::{ColorFormat};
pub use gfx::format::Depth;
use genmesh::{Vertices, Triangulate};
use genmesh::generators::{Plane, SharedVertex, IndexedPolygon};
use std::time::{Instant};

use gfx::Bundle;
//use allegro::{EventQueue, Core};
//use allegro::internal::events::{MouseAxes};
use winit::{Event, VirtualKeyCode, WindowBuilder};

pub type GFormat = [f32; 4];

gfx_defines!{
    vertex Vertex {
        pos: [f32; 3] = "a_Pos",
        color: [f32; 3] = "a_Color",
    }

    vertex BlitVertex {
        pos: [i8; 2] = "a_Pos",
        tex_coord: [i8; 2] = "a_TexCoord",
    }

    constant Locals {
        model: [[f32; 4]; 4] = "u_Model",
        view: [[f32; 4]; 4] = "u_View",
        proj: [[f32; 4]; 4] = "u_Proj",
        ipos: [f32; 4] = "u_ipos",
        cpos: [f32; 4] = "u_cpos",
        icolor: [f32; 4] = "u_color",
        iscale: f32 = "u_iscale",
    }

    pipeline pipe {
        vbuf: gfx::VertexBuffer<Vertex> = (),
        locals: gfx::ConstantBuffer<Locals> = "Locals",
        model: gfx::Global<[[f32; 4]; 4]> = "u_Model",
        view: gfx::Global<[[f32; 4]; 4]> = "u_View",
        proj: gfx::Global<[[f32; 4]; 4]> = "u_Proj",
        //out_color: gfx::RenderTarget<ColorFormat> = "Target0",
        out_color: gfx::RenderTarget<GFormat> = "Target0",
        //out_depth: gfx::DepthTarget<DepthFormat> =
        out_depth: gfx::DepthTarget<Depth> =
            gfx::preset::depth::LESS_EQUAL_WRITE,
    }

    pipeline blit {
        vbuf: gfx::VertexBuffer<BlitVertex> = (),
        tex: gfx::TextureSampler<[f32; 4]> = "t_BlitTex",
        out: gfx::RenderTarget<ColorFormat> = "Target0",
        locals: gfx::ConstantBuffer<Locals> = "Locals",
    }
}

fn calculate_color(height: f32) -> [f32; 3] {
    if height > 8.0 {
        [0.9, 0.9, 0.9] // white
    } else if height > 0.0 {
        [0.7, 0.7, 0.7] // greay
    } else if height > -5.0 {
        [0.2, 0.7, 0.2] // green
    } else {
        [0.2, 0.2, 0.7] // blue
    }
}


struct Terrain {
    tilelist : std::rc::Rc<std::cell::RefCell<Vec<(Vector3<f32>, f32, (f32,f32,f32))>>>
}
impl Terrain {
    fn new() -> Self {
        Terrain {tilelist: std::rc::Rc::new(std::cell::RefCell::new(Vec::new()))}
    }
    fn render_terrain(& self, depth: i32, origin: Vector3<f32>, cam_pos: Vector3<f32>, size: f32, rot: (f32,f32,f32), cam_mat: Matrix4<f32>) {

                use cgmath::{EuclideanVector, Matrix4, Rotation3, Vector, Matrix3, Vector4};


        let roti = (cgmath::Rad::from(cgmath::Deg{s:(rot.0)}),
                    cgmath::Rad::from(cgmath::Deg{s:(rot.1)}),
                    cgmath::Rad::from(cgmath::Deg{s:(rot.2)}));
        let final_height = 25.0*100.0*2.0;
        let mat_lif = cgmath::Matrix4::from_translation(Vector3::new(0.0,0.0,final_height));

        let mat_rot =  cgmath::Matrix4::from(cgmath:: Matrix3::from_euler(roti.0,roti.1,roti.2)) * mat_lif;

        let cam_pointer = cam_mat*cgmath::Vector4::new(0.0,0.0,1.0,1.0);
        let cam_pointer3 = Vector3::new(cam_pointer.x, cam_pointer.y, cam_pointer.z);
        //let cam_ray = cam_pointer3-Vector3::new(-self.cam_move.x, -self.cam_move.y, self.cam_move.z);
        let cam_ray = cam_pointer3-cam_pos;
        let cam_ray_n = cam_ray.normalize();


        let z2 = mat_rot * cgmath::Vector4::new(0.0,0.0,1.0, 1.0);
        let z = Vector3::new(z2.x, z2.y, z2.z);


        if cam_ray_n.dot(z.normalize()) >= -0.0 {
            return;
        }


            //let z = (origin).length() < 350.0 ;
            //println!("EVAL {:?}", z);

            let sss = 1.0*depth as f32 +1.0;
            let d = 1.0*25.0 * size;
            let bbss = 0.5;
                /*if depth <= 1
                {
                    self.render_terrain(depth+1, origin+Vector3::new(d,d,0.0), cam_pos, size * bbss);
                    self.render_terrain(depth+1, origin+Vector3::new(d,-d,0.0), cam_pos,size * bbss);
                    self.render_terrain(depth+1, origin+Vector3::new(-d,-d,0.0), cam_pos , size * bbss);
                    self.render_terrain(depth+1, origin+Vector3::new(-d,d,0.0), cam_pos, size * bbss);
                }
                else*/
                 if   depth < 2// best 6
                {
                    let mut idxofnearest = Vector3::<f32>::new(0.0,0.0,0.0);
                    let mut nearestdist = 99999999.0;
                    let z = [origin+Vector3::new(d,d,0.0),
                            origin+Vector3::new(d,-d,0.0),
                            origin+Vector3::new(-d,-d,0.0),
                            origin+Vector3::new(-d,d,0.0)];

                    for y in z.iter()
                    {
                        let vvvv = *y;
                        let z2 = mat_rot * cgmath::Vector4::new(vvvv.x,vvvv.y, vvvv.z, 1.0);
                        let z = Vector3::new(z2.x, z2.y, z2.z);
                        if (cam_pos-(z)).length() < 100.0*25.0*8.0/((1.0+depth as f32)*(1.0+depth as f32))
                        {
                            self.render_terrain(depth+1,*y, cam_pos, size * bbss, rot, cam_mat);
                        }
                        else
                        {
                            self.render_terrain(depth+100000,*y, cam_pos, size * bbss, rot, cam_mat);
                        }
                    }

                }

                else
                {
                    self.tilelist.borrow_mut().push((origin, size/bbss, rot));

                }


    }

    fn getlist(&self) -> &std::rc::Rc<std::cell::RefCell<Vec<(Vector3<f32>, f32, (f32,f32,f32))>>>
    {
        return &self.tilelist;
    }
}

struct ViewPair<R: gfx::Resources, T: gfx::format::Formatted> {
    resource: gfx::handle::ShaderResourceView<R, T::View>,
    target: gfx::handle::RenderTargetView<R, T>,
}

struct DepthFormat;
impl gfx::format::Formatted for DepthFormat {
    type Surface = gfx::format::D24;
    type Channel = gfx::format::Unorm;
    type View = [f32; 4];

    fn get_format() -> gfx::format::Format {
        use gfx::format as f;
        f::Format(f::SurfaceType::D24, f::ChannelType::Unorm)
    }
}

fn create_g_buffer<R: gfx::Resources, F: gfx::Factory<R>>(
                   width: gfx::texture::Size, height: gfx::texture::Size, factory: &mut F)
                   -> (ViewPair<R, GFormat>, ViewPair<R, GFormat>, ViewPair<R, GFormat>,
                       gfx::handle::ShaderResourceView<R, [f32; 4]>, gfx::handle::DepthStencilView<R, Depth>)
{
    use gfx::format::ChannelSource;
    let pos = {
        let (_ , srv, rtv) = factory.create_render_target(width, height).unwrap();
        ViewPair{ resource: srv, target: rtv }
    };
    let normal = {
        let (_ , srv, rtv) = factory.create_render_target(width, height).unwrap();
        ViewPair{ resource: srv, target: rtv }
    };
    let diffuse = {
        let (_ , srv, rtv) = factory.create_render_target(width, height).unwrap();
        ViewPair{ resource: srv, target: rtv }
    };
    let (tex, _srv, depth_rtv) = factory.create_depth_stencil(width, height).unwrap();
    // ignoring the default SRV since we need to create a custom one with swizzling
    let swizzle = gfx::format::Swizzle(ChannelSource::X, ChannelSource::X, ChannelSource::X, ChannelSource::X);
    let depth_srv = factory.view_texture_as_shader_resource::<DepthFormat>(&tex, (0,0), swizzle).unwrap();

    (pos, normal, diffuse, depth_srv, depth_rtv)
}


struct App<R: gfx::Resources> {
    //pso: gfx::PipelineState<R, pipe::Meta>,
    //data: pipe::Data<R>,
    //slice: gfx::Slice<R>,
    terrainP: Bundle<R, pipe::Data<R>>,
    start_time: Instant,
    terrain: Terrain,
    colors: Vec<(f32,f32,f32)>,
    cam_move: Vector3<f32>,
    cam_rot: (f32,f32),
    cam_mat: AffineMatrix3<f32>,
    inmanager:  multi_input::manager::RawInputManager,
    intermediate: ViewPair<R, GFormat>,
    debug_buf: Option<gfx::handle::ShaderResourceView<R, [f32; 4]>>,
    blit: Bundle<R, blit::Data<R>>,
}
/*
impl<R: gfx::Resources> gfx_app::Application<R> for App<R> {
    fn new<F: gfx::Factory<R>>(mut factory: F, init: gfx_app::Init<R>) -> Self {
    */
impl<R: gfx::Resources> gfx_app::Application<R> for App<R> {
fn new<F: gfx::Factory<R>>(factory: &mut F, backend: gfx_app::shade::Backend,
       window_targets: gfx_app::WindowTargets<R>) -> Self {
        use gfx::traits::FactoryExt;


        let (width, height, _, _) = window_targets.color.get_dimensions();
        let (gpos, gnormal, gdiffuse, _depth_resource, depth_target) =
            create_g_buffer(width, height, factory);
        let res = {
            let (_ , srv, rtv) = factory.create_render_target(width, height).unwrap();
            ViewPair{ resource: srv, target: rtv }
        };


        let sampler = factory.create_sampler(
            gfx::texture::SamplerInfo::new(gfx::texture::FilterMethod::Scale,
                                       gfx::texture::WrapMode::Clamp)
        );

        let blit = {
            let vertex_data = [
                BlitVertex { pos: [-3, -1], tex_coord: [-1, 0] },
                BlitVertex { pos: [ 1, -1], tex_coord: [1, 0] },
                BlitVertex { pos: [ 1,  3], tex_coord: [1, 2] },
            ];

            let (vbuf, slice) = factory.create_vertex_buffer_with_slice(&vertex_data, ());

            let vs = gfx_app::shade::Source {
                glsl_150: include_bytes!("shader/blit.glslv"),
                hlsl_40:  include_bytes!("data/blit_vs.fx"),
                .. gfx_app::shade::Source::empty()
            };
            let ps = gfx_app::shade::Source {
                glsl_150: include_bytes!("shader/blit.glslf"),
                hlsl_40:  include_bytes!("data/blit_ps.fx"),
                .. gfx_app::shade::Source::empty()
            };

            let pso = factory.create_pipeline_simple(
                vs.select(backend).unwrap(),
                ps.select(backend).unwrap(),
                blit::new()
                ).unwrap();

            let data = blit::Data {
                vbuf: vbuf,
                tex: (gpos.resource.clone(), sampler.clone()),
                out: window_targets.color,
                locals: factory.create_constant_buffer(1),
            };

            Bundle::new(slice, pso, data)
        };


        let mut ccc = Vec::new();
        for z in 0..100000
        {
            ccc.push(rand::random::<(f32, f32,f32)>());

        }
        let vs = gfx_app::shade::Source {
            glsl_120: include_bytes!("shader/terrain_120.glslv"),
            glsl_150: include_bytes!("shader/terrain_150.glslv"),
            hlsl_50:  include_bytes!("data/vertex.fx"),
            msl_11: include_bytes!("shader/terrain_vertex.metal"),
            vulkan:   include_bytes!("data/vert.spv"),
            .. gfx_app::shade::Source::empty()
        };
        let ps = gfx_app::shade::Source {
            glsl_120: include_bytes!("shader/terrain_120.glslf"),
            glsl_150: include_bytes!("shader/terrain_150.glslf"),
            hlsl_50:  include_bytes!("data/pixel.fx"),
            msl_11: include_bytes!("shader/terrain_frag.metal"),
            vulkan:   include_bytes!("data/frag.spv"),
            .. gfx_app::shade::Source::empty()
        };
        let ds = gfx_app::shade::Source {
            glsl_120: include_bytes!("shader/terrain_120.glslf"),
            glsl_150: include_bytes!("shader/terrain_150.glslf"),
            hlsl_50:  include_bytes!("data/domain.fx"),
            msl_11: include_bytes!("shader/terrain_frag.metal"),
            vulkan:   include_bytes!("data/frag.spv"),
            .. gfx_app::shade::Source::empty()
        };
        let hs = gfx_app::shade::Source {
            glsl_120: include_bytes!("shader/terrain_120.glslf"),
            glsl_150: include_bytes!("shader/terrain_150.glslf"),
            hlsl_50:  include_bytes!("data/hull.fx"),
            msl_11: include_bytes!("shader/terrain_frag.metal"),
            vulkan:   include_bytes!("data/frag.spv"),
            .. gfx_app::shade::Source::empty()
        };

        let rand_seed = rand::thread_rng().gen();
        let seed = Seed::new(rand_seed);
        //let plane = Plane::subdivide(256, 256);
        let plane = Plane::subdivide(128, 128); // 256 Works good last time, 32 min
        let vertex_data: Vec<Vertex> = plane.shared_vertex_iter()
            .map(|(x, y)| {
                let h = 0.0;
                Vertex {
                    pos: [25.0 * x, 25.0 * y, h*0.0],
                    color: calculate_color(h),
                }
            })
            .collect();

        let index_data: Vec<u32> = plane.indexed_polygon_iter()
            //.triangulate()
            .vertices()
            .map(|i| i as u32)
            .collect();

/*
        let mut index_data: Vec<u32> = Vec::new();
        index_data.push(0);
        index_data.push(10);
        index_data.push(20);
        index_data.push(30);
*/
        let (vbuf, slice) = factory.create_vertex_buffer_with_slice(&vertex_data, &index_data[..]);
        let aspect_ratio = width as f32 / height as f32;
        let data = pipe::Data {
            vbuf: vbuf,
            locals: factory.create_constant_buffer(1),
            model: Matrix4::identity().into(),
            view: Matrix4::identity().into(),
            proj: cgmath::perspective(
                cgmath::deg(60.0f32), aspect_ratio, 0.1, 50000.0
                ).into(),
            out_color: res.target.clone(),
            out_depth: depth_target.clone(),

        };
        let set = (factory.create_shader_set_tessellation(&vs.select(backend).unwrap(), &hs.select(backend).unwrap(), &ds.select(backend).unwrap(), &ps.select(backend).unwrap())).unwrap();

        let pso = factory.create_pipeline_state(&set, gfx::Primitive::PatchList(4), gfx::state::Rasterizer::new_fill(),
                               pipe::new()).unwrap();

        let terrain = Bundle::new(slice, pso, data);

        let cam_mat =  Transform::look_at(
            Point3::new(0.0 ,52.0, 30.0+16.0),//+10.0+480.0),
            Point3::new(0.0, 0.0, 40.0),//+450.0),
            Vector3::unit_z(),
        );
        use cgmath::{Vector, Vector3};

        let mut manager = multi_input::manager::RawInputManager::new().unwrap();
        manager.register_devices(multi_input::DeviceType::Joysticks);
        //manager.register_devices(multi_input::DeviceType::Keyboards);
        manager.register_devices(multi_input::DeviceType::Mice);
        //let set = try!(factory.create_shader_set(&vs, &ps));


        App {

            /*pso: factory.create_pipeline_state(&set, gfx::Primitive::QuadList, gfx::state::Rasterizer::new_fill(),
                                   pipe::new()).unwrap(),*/
              /*factory.create_pipeline_simple(
                vs.select(init.backend).unwrap(),
                ps.select(init.backend).unwrap(),
                pipe::new()
            ).unwrap(),*/
            //data: ,
            terrainP: terrain,
            //slice: slice,
            start_time: Instant::now(),
            terrain: Terrain::new(),
            colors: ccc,
            cam_mat: cam_mat,
            cam_move: Vector3::new(0.0, -9000.0,  -1000.0),
            cam_rot: (0.0,0.0),
            inmanager: manager,
            intermediate: res,
            debug_buf: None,
            blit: blit,
        }
    }



    fn render<C: gfx::CommandBuffer<R>>(&mut self, encoder: &mut gfx::Encoder<R, C>) {
        let elapsed = self.start_time.elapsed();
        let time = (elapsed.as_secs() as f32 + elapsed.subsec_nanos() as f32 / 1000_000_000.0)*0.2;
        let x = time.sin();
        let y = time.cos();
        /*
        let view: AffineMatrix3<f32> = Transform::look_at(
            Point3::new(x * 52.0, y * 52.0, 30.0+16.0),//+10.0+480.0),
            Point3::new(0.0, 0.0, 40.0),//+450.0),
            Vector3::unit_z(),
        );*/
        //self.cam_mat.position;
        use cgmath::{Vector, Vector3, EuclideanVector, Point3, Point, Rad};
        let rot_a = cgmath::Matrix4::from(cgmath::Matrix3::from_euler(cgmath::Rad{s:(0.0)},cgmath::Rad{s:(self.cam_rot.0*0.1)}, cgmath::Rad{s:(self.cam_rot.0*0.0)}));
        let rot_b = cgmath::Matrix4::from(cgmath::Matrix3::from_euler(cgmath::Rad{s:(self.cam_rot.1*0.1)},cgmath::Rad{s:(self.cam_rot.0*0.0)}, cgmath::Rad{s:(self.cam_rot.0*0.0)}));
        self.cam_mat.mat  =   rot_a* rot_b * self.cam_mat.mat ;
                            ;

        let view = self.cam_mat.mat  * cgmath::Matrix4::from_translation(self.cam_move);

        let final_height = 25.0*100.0*2.0;
        //self.terrain.render_terrain(0, Vector3::new(0.0,0.0,0.0),Vector3::new(-self.cam_move.x, -self.cam_move.y, -self.cam_move.z), 100.0);
        self.terrain.tilelist.borrow_mut().clear();

        let worldbox= [(0.0, 0.0, 0.0),
                        (0.0, 90.0, 0.0),
                        (0.0, -90.0, 0.0),
                        (0.0, 180.0, 0.0),
                        (-90.0, 0.0, 0.0),
                        (90.0, 0.0, 0.0)];
        //if depth == 0
        {
        //    self.tilelist.borrow_mut().clear();
        }
        for t_rot in worldbox.iter()
        {
            self.terrain.render_terrain(0, Vector3::new(0.0,0.0,0.0),
                                            Vector3::new(-self.cam_move.x, -self.cam_move.y, -self.cam_move.z), 100.0,
                                        *t_rot, self.cam_mat.mat);
        }


        encoder.clear(&self.terrainP.data.out_color, [0.0, 0.0, 0.0, 1.0]);
        encoder.clear_depth(&self.terrainP.data.out_depth, 1.0);


        self.terrainP.data.view = (view).into();
/*
        let locals = Locals {
            model: self.data.model,
            view: self.data.view,
            proj: self.data.proj,
            ipos: [0.0,0.0,0.0],
            iscale: 1.0
        };

        encoder.update_buffer(&self.data.locals, &[locals], 0).unwrap();
        encoder.clear(&self.data.out_color, [1.0, 0.0, 0.0, 1.0]);
        encoder.clear_depth(&self.data.out_depth, 1.0);
        encoder.draw(&self.slice, &self.pso, &self.data);

        let locals = Locals {
            model: self.data.model,
            view: self.data.view,
            proj: self.data.proj,
            ipos: [50.0*0.75,25.0*0.5,0.0],
            iscale: 0.5
        };

        encoder.update_buffer(&self.data.locals, &[locals], 0).unwrap();
        encoder.draw(&self.slice, &self.pso, &self.data);

        let locals = Locals {
            model: self.data.model,
            view: self.data.view,
            proj: self.data.proj,
            ipos: [50.0*0.75,-25.0*0.5,0.0],
            iscale: 0.5
        };

        encoder.update_buffer(&self.data.locals, &[locals], 0).unwrap();
        encoder.draw(&self.slice, &self.pso, &self.data);*/

println!("{:?}", self.cam_move);
        let mylist = self.terrain.getlist().borrow();
        println!("{:?}", mylist.len());
        let mut patch_counter  = 0;




        let blit_tex = match self.debug_buf {
            Some(ref tex) => tex,   // Show one of the immediate buffers
            None => {

                encoder.clear(&self.intermediate.target, [135.0/256.0, 206.0/256.0, 250.0/256.0, 1.0]);
                // Apply lights
                //self.light.encode(encoder);
                // Draw light emitters
                //self.emitter.encode(encoder);
                for (idx,tup) in mylist.iter().enumerate()
                {
                    ;
                    use cgmath::{EuclideanVector, Matrix4, Rotation3, Vector, Matrix3, Vector4};
                    let cam_pointer = self.cam_mat.mat*Vector4::new(0.0,0.0,-1.0,1.0);
                    let cam_pointer3 = Vector3::new(cam_pointer.x, cam_pointer.y, cam_pointer.z);
                    //let cam_ray = cam_pointer3-Vector3::new(-self.cam_move.x, -self.cam_move.y, self.cam_move.z);
                    let cam_ray = cam_pointer3-self.cam_move;
                    let cam_ray_n = cam_ray.normalize();
                    if cam_ray_n.dot((tup.0-self.cam_move).normalize()) < -0.0 {
                        continue;
                    }
                    let roti = (cgmath::Rad::from(cgmath::Deg{s:((tup.2).0)}),
                                cgmath::Rad::from(cgmath::Deg{s:(tup.2).1}),
                                cgmath::Rad::from(cgmath::Deg{s:(tup.2).2}));
                    let mat_rot = Matrix4::from( Matrix3::from_euler(roti.0,roti.1,roti.2));
                    let mat_tra = Matrix4::from_translation(tup.0);
                    let mat_lif = Matrix4::from_translation(Vector3::new(0.0,0.0,final_height));
                    let color = self.colors[idx];
                    let locals = Locals {
                        model:  (mat_rot*mat_tra*mat_lif ).into(),
                        view: self.terrainP.data.view,
                        proj: self.terrainP.data.proj,
                        ipos: [tup.0.x, tup.0.y, tup.0.z, 0.0],
                        cpos: [self.cam_move.x, self.cam_move.y, self.cam_move.z, 0.0],
                        iscale: tup.1,
                        icolor: [color.0, color.1, color.2, 0.0]
                    };
                    patch_counter += 1;
                    encoder.update_constant_buffer(&self.terrainP.data.locals, &locals);
                    encoder.update_buffer(&self.terrainP.data.locals, &[locals], 0);//.unwrap();
                    //encoder.draw(&self.slice, &self.pso, &self.data);
                    //self.terrainP.encode(encoder);
                    self.terrainP.encode(encoder);

                }




                &self.intermediate.resource
            }
        };
        self.blit.data.tex.0 = blit_tex.clone();
        // Show the result

        let locals = Locals {
            model:  Matrix4::identity().into(),
            view: self.terrainP.data.view,
            proj: self.terrainP.data.proj,
            ipos: [0.0, 0.0, 0.0, 0.0],
            cpos: [self.cam_move.x, self.cam_move.y, self.cam_move.z, 0.0],
            iscale: 1.0,
            icolor: [0.0, 0.0, 0.0, 0.0]
        };
        patch_counter += 1;
        encoder.update_constant_buffer(&self.blit.data.locals, &locals);


        self.blit.encode(encoder);

        println!("Patches {:?}",patch_counter );
        //let s = winapi::xinput::XINPUT_KEYSTROKE {};
        //winapi::xinput::XInputGetKeystroke(winapi::xinput::XUSER_INDEX_ANY, (),  &s)
        //use multiinput::*;


        let mut zzz = (self.cam_move.length()-4800.0);
        if zzz < 0.0
            {
                zzz = -zzz;
            }

            let move_speed = 3.0*1.0*zzz;
        let timedelta = time*0.01;

        //self.cam_move = Vector3::zero();
        self.cam_rot = (0.0, 0.0);
        //let keys = (self.eventpump.keyboard_state().pressed_scancodes().filter_map(Keycode::from_scancode));//.collect<Option<Keycode::Keycode>>();
        //let keysl: HashSet<Keycode> = keys.collect();
        //println!("{:?}",  keys);
        /*
        loop {
            let e = self.eventpump.poll_event();
            if e.is_none()
                {continue;}

            let e2 = e.unwrap();
            match e2 {
                Event::KeyDown {..} => println!("{:?}", "YAHOOOOOOOOOOOOOOOOOOOOOOO"),
                _ => ()
            }
        }
        if self.eventpump.keyboard_state().is_scancode_pressed(sdl2::keyboard::Scancode::Q)
        {
            print!("{:?}", "Q is pressed......................................................");
        }*/
use multi_input::*;

for x in 0..50 {
        let event2 = self.inmanager.get_event();
        if event2.is_some()
        {
            let event = event2.unwrap();
            match event{
                RawEvent::MouseMoveEvent(_, a, b)
                    => self.cam_rot =(timedelta * a as f32,  timedelta * b as f32 ),
                _ => (),
            }
        }
}
        use sfml::window::{Key};



        if Key::W.is_pressed() {
            self.cam_move = self.cam_move +  self.cam_mat.transform_vector(Vector3::unit_z()) * timedelta * move_speed ;
        }

        if Key::S.is_pressed() {
             self.cam_move = self.cam_move - self.cam_mat.transform_vector(Vector3::unit_z()) * timedelta * move_speed ;
        }

        if Key::A.is_pressed() {
            self.cam_move = self.cam_move + self.cam_mat.transform_vector(Vector3::unit_x()) * timedelta * move_speed;
        }

        if Key::D.is_pressed() {
            self.cam_move = self.cam_move - self.cam_mat.transform_vector(Vector3::unit_x()) * timedelta * move_speed;
        }

    }
}

pub fn main() {
use gfx_app::Application;
   let wb = WindowBuilder::new().with_dimensions(1366, 768).with_title("Rusty Planet demo");
   App::launch_default(wb);
}
