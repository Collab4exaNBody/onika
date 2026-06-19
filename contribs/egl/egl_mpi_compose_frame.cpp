
/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/

#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>
#include <onika/log.h>
#include <EGLRender/egl_render_manager.h>
#include <mpi.h>

namespace OnikaEGLRender
{

  using namespace onika;
  using namespace onika::scg;
  using namespace EGLRender;

  class EGLRenderSurfaceMPICompose : public OperatorNode
  {
    ADD_SLOT( MPI_Comm , mpi , INPUT_OUTPUT , MPI_COMM_WORLD );
    ADD_SLOT( std::string , surface        , INPUT , "window" );
    ADD_SLOT( EGLRenderManager , egl_render_manager , INPUT_OUTPUT );

  public:
    inline bool is_sink() const override final { return true; }

    inline void execute() override final
    {
      int sufrace_id = egl_render_manager->surface_id( *surface );
      if( sufrace_id < 0 )
      {
        lerr << "EGL surface "<< *surface << " not found"<<std::endl;
        return;
      }
      auto & surf = egl_render_manager->surface(sufrace_id);
      long width = surf.width();
      long height = surf.height();

      int nproc = 1;
      int rank = 0;
      MPI_Comm_rank(*mpi,&rank);
      MPI_Comm_size(*mpi,&nproc);

      if( nproc <= 1 ) return;

      //const long alloc_pixel_sz = width * (height+nproc);
      const long comm_hrange = (height+nproc-1) / nproc;
      const long comm_pixel_sz = comm_hrange * width;
      const long alloc_comm_pixels = comm_pixel_sz * ( nproc + 1 ); // + 1 to reserve space for composition

      int read_color_buffer_id = egl_render_manager->create_pixel_buffer( "mpi_compose_read_color_buffer" , width, height + nproc, GL_RGBA, GL_PIXEL_PACK_BUFFER );
      auto & read_color_buffer = egl_render_manager->pixel_buffer(read_color_buffer_id);
      read_color_buffer.read_pixels();
      read_color_buffer.unuse();
      const uint32_t * pixel_data = (const uint32_t*) read_color_buffer.map_buffer_read_only();

      int read_depth_buffer_id = egl_render_manager->create_pixel_buffer( "mpi_compose_read_depth_buffer" , width, height + nproc, GL_DEPTH_COMPONENT, GL_PIXEL_PACK_BUFFER );
      auto & read_depth_buffer = egl_render_manager->pixel_buffer(read_depth_buffer_id);
      read_depth_buffer.read_pixels();
      read_depth_buffer.unuse();
      const GLfloat * depth_data = (const GLfloat*) read_depth_buffer.map_buffer_read_only();

      // communication and composition scratch space
      auto other_pixel_data = std::make_unique_for_overwrite<uint32_t[]>( alloc_comm_pixels );
      auto other_depth_data = std::make_unique_for_overwrite<GLfloat []>( alloc_comm_pixels );

      auto mpi_rank_to_height_range = [height](long rank, long nproc) -> std::pair<long,long>
      {
        return { (height*rank)/nproc , (height*(rank+1))/nproc };
      };

      for(int p=0;p<nproc;p++)
      {
        const auto [hstart,hend] = mpi_rank_to_height_range(p,nproc);
        //const long hsize = hend - hstart;
        MPI_Gather( pixel_data + ( hstart * width ) , comm_pixel_sz , MPI_UNSIGNED , other_pixel_data.get() , comm_pixel_sz , MPI_UNSIGNED , p , *mpi );
        MPI_Gather( depth_data + ( hstart * width ) , comm_pixel_sz , MPI_FLOAT , other_depth_data.get() , comm_pixel_sz , MPI_FLOAT , p , *mpi );
      }
      read_color_buffer.unmap_buffer();
      read_depth_buffer.unmap_buffer();

      const auto [comp_hstart,comp_hend] = mpi_rank_to_height_range(rank,nproc);
      const auto comp_hsize = comp_hend - comp_hstart;
      uint32_t * comp_d_pixels = other_pixel_data.get() + comm_pixel_sz * nproc;
      GLfloat * comp_d_depth = other_depth_data.get() + comm_pixel_sz * nproc;

      std::memcpy( comp_d_pixels , other_pixel_data.get() + comm_pixel_sz * rank , comp_hsize * width * 4 );
      std::memcpy( comp_d_depth , other_depth_data.get() + comm_pixel_sz * rank , comp_hsize * width * 4 );

      long composed_pixel_count = 0;
      for(int p=0;p<nproc;p++) if( p != rank )
      {
        const auto * s_pixels = other_pixel_data.get() + comm_pixel_sz * p;
        const auto * s_depth = other_depth_data.get() + comm_pixel_sz * p;
        //const long hsize = hend - hstart;
//#       pragma omp parallel for schedule(static)
        for(long y=0;y<comp_hsize;y++)
        {
//#         pragma omp simd
          for(long x=0;x<width;x++)
          {
            const long i = y*width + x;
            if( s_depth[i] < comp_d_depth[i] )
            {
              ++ composed_pixel_count;
              comp_d_depth[i] = s_depth[i];
              comp_d_pixels[i] = s_pixels[i];
            }
          }
        }
      }

      ldbg <<"merged "<<composed_pixel_count<<" pixels"<<std::endl;

      MPI_Gather( comp_d_pixels , comm_pixel_sz , MPI_UNSIGNED , other_pixel_data.get() , comm_pixel_sz , MPI_UNSIGNED , 0 , *mpi );

      if( rank == 0 )
      {
        int write_pixel_buffer_id = egl_render_manager->create_pixel_buffer("mpi_compose_write_pixel_buffer",width,height,GL_RGBA,GL_PIXEL_UNPACK_BUFFER);
        auto & write_pixel_buffer = egl_render_manager->pixel_buffer(write_pixel_buffer_id);
        uint32_t * out_pixel_data = (uint32_t*) write_pixel_buffer.map_buffer_write_only();
        for(int p=0;p<nproc;p++)
        {
          const auto [hstart,hend] = mpi_rank_to_height_range(p,nproc);
          const auto hsize = hend - hstart;
          const uint32_t * s_pixels = other_pixel_data.get() + comm_pixel_sz * p;
          uint32_t * d_pixels = out_pixel_data + hstart * width;
          std::memcpy( d_pixels , s_pixels , hsize * width * 4 );
        }
        write_pixel_buffer.unmap_buffer();
        write_pixel_buffer.copy_to_texture();

        int fb_id = egl_render_manager->frame_buffer_id("mpi_compose_framebuffer");
        if( fb_id < 0 )
        {
          fb_id = egl_render_manager->create_frame_buffer("mpi_compose_framebuffer" , GL_READ_FRAMEBUFFER );
          auto & fb = egl_render_manager->frame_buffer(fb_id);
          fb.bind();
          fb.attach_texture( write_pixel_buffer.copy_to_texture() , GL_COLOR_ATTACHMENT0 );
          fb.unbind();
        }
        auto & fb = egl_render_manager->frame_buffer(fb_id);
        fb.bind();
        glBlitFramebuffer(0,0,width,height,0,0,width,height,GL_COLOR_BUFFER_BIT,GL_NEAREST);
        fb.unbind();
      }
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_mpi_compose_frame)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_mpi_compose_frame", make_compatible_operator< EGLRenderSurfaceMPICompose > );
  }

}

