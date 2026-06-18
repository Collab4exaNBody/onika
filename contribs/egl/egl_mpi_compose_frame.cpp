
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
      auto & surf = egl_render_manager->surface(*surface);
      long width = surf.width();
      long height = surf.height();

      int nproc = 1;      
      int rank = 0;
      MPI_Comm_rank(*mpi,&rank);
      MPI_Comm_size(*mpi,&nproc);

      auto mpi_rank_to_height_range = [height](long rank, long nproc) -> std::pair<long,long>
      {
        return { (height*rank)/nproc , (height*(rank+1))/nproc };
      };

      const long alloc_pixel_sz = width * (height+nproc);
      const long comm_hrange = (height+nproc-1) / nproc;
      const long comm_pixel_sz = comm_hrange * width;
      const long alloc_comm_pixels = comm_pixel_sz * nproc;

      auto pixel_data = std::make_unique_for_overwrite<uint32_t[]>(alloc_pixel_sz);
      glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixel_data.get() );

      auto depth_data = std::make_unique_for_overwrite<GLfloat[]>(alloc_pixel_sz);
      glReadPixels(0, 0, width, height,  GL_DEPTH_COMPONENT , GL_FLOAT, depth_data.get() );   
      
      auto other_pixel_data = std::make_unique_for_overwrite<uint32_t[]>( alloc_comm_pixels );
      auto other_depth_data = std::make_unique_for_overwrite<GLfloat []>( alloc_comm_pixels );

      for(int p=0;p<nproc;p++)
      {
        const auto [hstart,hend] = mpi_rank_to_height_range(p,nproc);
        //const long hsize = hend - hstart;
        MPI_Gather( pixel_data.get() + ( hstart * width ) , comm_pixel_sz , MPI_UNSIGNED , other_pixel_data.get() , comm_pixel_sz , MPI_UNSIGNED , p , *mpi );
        MPI_Gather( depth_data.get() + ( hstart * width ) , comm_pixel_sz , MPI_UNSIGNED , other_depth_data.get() , comm_pixel_sz , MPI_UNSIGNED , p , *mpi );
      }

      const auto [comp_hstart,comp_hend] = mpi_rank_to_height_range(rank,nproc);
      const auto comp_hsize = comp_hend - comp_hstart;
      auto * comp_d_pixels = pixel_data.get() + comp_hstart * width;
      auto * comp_d_depth = depth_data.get() + comp_hstart * width;

      long composed_pixel_count = 0;
      for(int p=0;p<nproc;p++)
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
      
      ldbg <<"inserted "<<composed_pixel_count<<" pixels"<<std::endl;

      MPI_Gather( comp_d_pixels , comm_pixel_sz , MPI_UNSIGNED , other_pixel_data.get() , comm_pixel_sz , MPI_UNSIGNED , 0 , *mpi );
      
      if( rank == 0 )
      {
        for(int p=0;p<nproc;p++)
        {
          const auto [hstart,hend] = mpi_rank_to_height_range(p,nproc);
          const auto hsize = hend - hstart;
          const auto * s_pixels = other_pixel_data.get() + comm_pixel_sz * p;
          auto * d_pixels = pixel_data.get() + hstart * width;
          std::memcpy( d_pixels , s_pixels , hsize * width );
          std::memset( d_pixels, 0xFF , hsize * width );
        }
        
        for(int i=0;i<(width*height);i++)
        {
          
        }
        glDisable(GL_BLEND);
        glDisable(GL_DEPTH);
        glDrawPixels( width, height/2, GL_RGBA, GL_UNSIGNED_BYTE, pixel_data.get() );
      }
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_mpi_compose_frame)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_mpi_compose_frame", make_compatible_operator< EGLRenderSurfaceMPICompose > );
  }

}

