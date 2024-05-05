#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <torch/extension.h>
#include "common.h"

inline __device__ void fetch2sharedBack(
    int32_t n,
    const int2 range,
    const int *__restrict__ gs_id_per_patch,
    const float *__restrict__ us,
    const float *__restrict__ cov2d_inv,
    const float *__restrict__ alphas,
    const float *__restrict__ colors,
    float2 *shared_pos2d,
    float3 *shared_cinv2d,
    float *shared_alpha,
    float3 *shared_color,
    int *shared_gsid)
{
    int i = blockDim.x * threadIdx.y + threadIdx.x;  // block idx
    int j = range.y - n * BLOCK_SIZE + i;  // patch idx
    if (j >= range.x)
    {
        int gs_id = gs_id_per_patch[j];
        shared_gsid[i] = gs_id;
        shared_pos2d[i].x = us[gs_id * 2];
        shared_pos2d[i].y = us[gs_id * 2 + 1];
        shared_cinv2d[i].x = cov2d_inv[gs_id * 3];
        shared_cinv2d[i].y = cov2d_inv[gs_id * 3 + 1];
        shared_cinv2d[i].z = cov2d_inv[gs_id * 3 + 2];
        shared_alpha[i] =   alphas[gs_id];
        shared_color[i].x = colors[gs_id * 3];
        shared_color[i].y = colors[gs_id * 3 + 1];
        shared_color[i].z = colors[gs_id * 3 + 2];
    }
}

__global__ void  drawBack __launch_bounds__(BLOCK * BLOCK)(
    const int W,
    const int H,
    const int *__restrict__ patch_offset_per_tile,
    const int *__restrict__ gs_id_per_patch,
    const float *__restrict__ us,
    const float *__restrict__ cov2d_inv,
    const float *__restrict__ alphas,
    const float *__restrict__ colors,
    const int *__restrict__ contrib,
    const float *__restrict__ final_tau,
    const float *__restrict__ dloss_dgammas,
    float *__restrict__ dloss_dalphas)

{
    const uint2 tile = {blockIdx.x, blockIdx.y};
    const uint2 pix = {tile.x * BLOCK + threadIdx.x,
                       tile.y * BLOCK + threadIdx.y};

    const int tile_idx = tile.y * gridDim.x + tile.x;
    const uint32_t pix_idx = W * pix.y + pix.x;

	const bool inside = pix.x < W && pix.y < H;
	const int2 range = {patch_offset_per_tile[tile_idx], 
                        patch_offset_per_tile[tile_idx] + contrib[pix_idx]};


	bool thread_is_finished = !inside;

	__shared__ float2 shared_pos2d[BLOCK_SIZE];
	__shared__ float3 shared_cinv2d[BLOCK_SIZE];
    __shared__ float  shared_alpha[BLOCK_SIZE];
    __shared__ float3 shared_color[BLOCK_SIZE];
    __shared__ int shared_gsid[BLOCK_SIZE];

	const int gs_num = range.y - range.x;

    float3 gamma_next = {0, 0, 0}; // final color of pix

    float3 dloss_dgamma = {dloss_dgammas[0 * H * W + pix_idx],
                           dloss_dgammas[1 * H * W + pix_idx],
                           dloss_dgammas[2 * H * W + pix_idx]};

    float tau = final_tau[pix_idx];

    int cont = 0;

    // for all 2d gaussian 
    for (int i = 0; i < gs_num; i++)
    {
        int finished_thread_num = __syncthreads_count(thread_is_finished);

        if (finished_thread_num == BLOCK_SIZE)
            break;

        int j = i % BLOCK_SIZE;

        if (j == 0)
        {
            // fetch 2d gaussian data to share memory
            // fetch to shared memory by backward order 
            fetch2sharedBack(i / BLOCK_SIZE,
                         range,
                         gs_id_per_patch,
                         us,
                         cov2d_inv,
                         alphas,
                         colors,
                         shared_pos2d,
                         shared_cinv2d,
                         shared_alpha,
                         shared_color,
                         shared_gsid);
            __syncthreads();
        }

        float2 u = shared_pos2d[j];
        float3 cinv = shared_cinv2d[j];
        float alpha = shared_alpha[j];
        float3 color = shared_color[j];
        int gs_id = shared_gsid[j];
        float2 d = u - pix;
        float maha_dist = max(0.0f,  mahaSqDist(cinv, d));
        float alpha_prime = min(0.99f, alpha * exp( -0.5f * maha_dist));

        float3 dgamma_dalphaprime = tau * (color - gamma_next);
        float dalphaprime_dalpha = exp(-0.5f * maha_dist);
        float dloss_dalpha = dot(dloss_dgamma, dgamma_dalphaprime) * dalphaprime_dalpha;
        printf("%d %f\n", gs_id, dloss_dalpha);
        // atomicAdd(&dloss_dalphas[gs_id], 0);
        // printf("%d %f\n", gs_id, dloss_dalphas[gs_id]);

        /*

        float3 gamma = alpha_prime * color + (1 - alpha_prime) * gamma_next;
        float tau_prev = tau / (1 - alpha_prime);

        //if ()


        

        // float dloss_dalpahprime = dot(dloss_dgamma, dgamma_dalphaprime);



        // forward.md (5.1)
        // mahalanobis squared distance for 2d gaussian to this pix
        
        float maha_dist = max(0.0f,  mahaSqDist(cinv, d));

        float alpha_prime = min(0.99f, alpha * exp( -0.5f * maha_dist));

        if (alpha_prime < 0.002f)
            continue;

        // forward.md (5)
        finial_color +=  tau * alpha_prime * color;
        cont = cont + 1;  // how many gs contribute to this pixel. 

        // forward.md (5.2)
        float tau_new = tau * (1.f - alpha_prime);

        if (tau_new < 0.0001f)
        {
            thread_is_finished = true;
            continue;
        }
        tau = tau_new;
        */
    }

    //if (inside)
    //{
    //    image[H * W * 0 + pix_idx] = finial_color.x;
    //    image[H * W * 1 + pix_idx] = finial_color.y;
    //    image[H * W * 2 + pix_idx] = finial_color.z;
    //    contrib[pix_idx] = cont;
    //    final_tau[pix_idx] = tau;
    //}
}


std::vector<torch::Tensor> backward(
    const int H,
    const int W,
    const torch::Tensor us,
    const torch::Tensor cov2d,
    const torch::Tensor alphas,
    const torch::Tensor depths,
    const torch::Tensor colors,
    const torch::Tensor contrib,
    const torch::Tensor final_tau, 
    const torch::Tensor patch_offset_per_tile, 
    const torch::Tensor gs_id_per_patch,
    const torch::Tensor cov2d_inv,
    const torch::Tensor dloss_dgammas)
{
    int gs_num = us.sizes()[0]; 
    dim3 grid(DIV_ROUND_UP(W, BLOCK), DIV_ROUND_UP(H, BLOCK), 1);
	dim3 block(BLOCK, BLOCK, 1);
    
    auto float_opts = us.options().dtype(torch::kFloat32);
    auto int_opts = us.options().dtype(torch::kInt32);
    torch::Tensor image = torch::full({3, H, W}, 0.0, float_opts);
    torch::Tensor dloss_dalphas = torch::full({gs_num}, 0, float_opts);

    
    drawBack<<<grid, block>>>(
        W,
        H,
        patch_offset_per_tile.contiguous().data_ptr<int>(),
        gs_id_per_patch.contiguous().data_ptr<int>(),
        us.contiguous().data_ptr<float>(),
        cov2d_inv.contiguous().data_ptr<float>(),
        alphas.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        contrib.contiguous().data_ptr<int>(),
        final_tau.contiguous().data_ptr<float>(),
        dloss_dgammas.contiguous().data_ptr<float>(),
        dloss_dalphas.contiguous().data_ptr<float>());
    
    cudaDeviceSynchronize();

   return {image};
    
    


}
