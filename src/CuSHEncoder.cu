#include "CuSHEncoder.h"


__global__ void CuSHKernel(
	const uint32_t num_elements,
	const uint32_t degree,
	float * data_in,
	float * data_out
){
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	data_out = data_out + (degree * degree) * i;

	float x = data_in[i * 3];
	float y = data_in[i * 3 + 1];
	float z = data_in[i * 3 + 2];

	// Let compiler figure out how to sequence/reorder these calculations w.r.t. branches
	float xy=x*y, xz=x*z, yz=y*z, x2=x*x, y2=y*y, z2=z*z;
	float x4=x2*x2, y4=y2*y2, z4=z2*z2;
	float x6=x4*x2, y6=y4*y2, z6=z4*z2;

	auto fill_sh = [&]() 
	{
		data_out[0] = 0.28209479177387814f;
		if (degree <= 1) { return; }

		data_out[1] = -0.48860251190291987f*y;
		data_out[2] = 0.48860251190291987f*z;
		data_out[3] = -0.48860251190291987f*x;
		if (degree <= 2) { return; }

		data_out[4] = 1.0925484305920792f*xy;
		data_out[5] = -1.0925484305920792f*yz;
		data_out[6] = 0.94617469575755997f*z2 - 0.31539156525251999f;
		data_out[7] = -1.0925484305920792f*xz;
		data_out[8] = 0.54627421529603959f*x2 - 0.54627421529603959f*y2;
		if (degree <= 3) { return; }

		data_out[9] = 0.59004358992664352f*y*(-3.0f*x2 + y2);
		data_out[10] = 2.8906114426405538f*xy*z;
		data_out[11] = 0.45704579946446572f*y*(1.0f - 5.0f*z2);
		data_out[12] = 0.3731763325901154f*z*(5.0f*z2 - 3.0f);
		data_out[13] = 0.45704579946446572f*x*(1.0f - 5.0f*z2);
		data_out[14] = 1.4453057213202769f*z*(x2 - y2);
		data_out[15] = 0.59004358992664352f*x*(-x2 + 3.0f*y2);
		if (degree <= 4) { return; }

		data_out[16] = 2.5033429417967046f*xy*(x2 - y2);
		data_out[17] = 1.7701307697799304f*yz*(-3.0f*x2 + y2);
		data_out[18] = 0.94617469575756008f*xy*(7.0f*z2 - 1.0f);
		data_out[19] = 0.66904654355728921f*yz*(3.0f - 7.0f*z2);
		data_out[20] = -3.1735664074561294f*z2 + 3.7024941420321507f*z4 + 0.31735664074561293f;
		data_out[21] = 0.66904654355728921f*xz*(3.0f - 7.0f*z2);
		data_out[22] = 0.47308734787878004f*(x2 - y2)*(7.0f*z2 - 1.0f);
		data_out[23] = 1.7701307697799304f*xz*(-x2 + 3.0f*y2);
		data_out[24] = -3.7550144126950569f*x2*y2 + 0.62583573544917614f*x4 + 0.62583573544917614f*y4;
		if (degree <= 5) { return; }

		data_out[25] = 0.65638205684017015f*y*(10.0f*x2*y2 - 5.0f*x4 - y4);
		data_out[26] = 8.3026492595241645f*xy*z*(x2 - y2);
		data_out[27] = -0.48923829943525038f*y*(3.0f*x2 - y2)*(9.0f*z2 - 1.0f);
		data_out[28] = 4.7935367849733241f*xy*z*(3.0f*z2 - 1.0f);
		data_out[29] = 0.45294665119569694f*y*(14.0f*z2 - 21.0f*z4 - 1.0f);
		data_out[30] = 0.1169503224534236f*z*(-70.0f*z2 + 63.0f*z4 + 15.0f);
		data_out[31] = 0.45294665119569694f*x*(14.0f*z2 - 21.0f*z4 - 1.0f);
		data_out[32] = 2.3967683924866621f*z*(x2 - y2)*(3.0f*z2 - 1.0f);
		data_out[33] = -0.48923829943525038f*x*(x2 - 3.0f*y2)*(9.0f*z2 - 1.0f);
		data_out[34] = 2.0756623148810411f*z*(-6.0f*x2*y2 + x4 + y4);
		data_out[35] = 0.65638205684017015f*x*(10.0f*x2*y2 - x4 - 5.0f*y4);
		if (degree <= 6) { return; }

		data_out[36] = 1.3663682103838286f*xy*(-10.0f*x2*y2 + 3.0f*x4 + 3.0f*y4);
		data_out[37] = 2.3666191622317521f*yz*(10.0f*x2*y2 - 5.0f*x4 - y4);
		data_out[38] = 2.0182596029148963f*xy*(x2 - y2)*(11.0f*z2 - 1.0f);
		data_out[39] = -0.92120525951492349f*yz*(3.0f*x2 - y2)*(11.0f*z2 - 3.0f);
		data_out[40] = 0.92120525951492349f*xy*(-18.0f*z2 + 33.0f*z4 + 1.0f);
		data_out[41] = 0.58262136251873131f*yz*(30.0f*z2 - 33.0f*z4 - 5.0f);
		data_out[42] = 6.6747662381009842f*z2 - 20.024298714302954f*z4 + 14.684485723822165f*z6 - 0.31784601133814211f;
		data_out[43] = 0.58262136251873131f*xz*(30.0f*z2 - 33.0f*z4 - 5.0f);
		data_out[44] = 0.46060262975746175f*(x2 - y2)*(11.0f*z2*(3.0f*z2 - 1.0f) - 7.0f*z2 + 1.0f);
		data_out[45] = -0.92120525951492349f*xz*(x2 - 3.0f*y2)*(11.0f*z2 - 3.0f);
		data_out[46] = 0.50456490072872406f*(11.0f*z2 - 1.0f)*(-6.0f*x2*y2 + x4 + y4);
		data_out[47] = 2.3666191622317521f*xz*(10.0f*x2*y2 - x4 - 5.0f*y4);
		data_out[48] = 10.247761577878714f*x2*y4 - 10.247761577878714f*x4*y2 + 0.6831841051919143f*x6 - 0.6831841051919143f*y6;
		if (degree <= 7) { return; }

		data_out[49] = 0.70716273252459627f*y*(-21.0f*x2*y4 + 35.0f*x4*y2 - 7.0f*x6 + y6);
		data_out[50] = 5.2919213236038001f*xy*z*(-10.0f*x2*y2 + 3.0f*x4 + 3.0f*y4);
		data_out[51] = -0.51891557872026028f*y*(13.0f*z2 - 1.0f)*(-10.0f*x2*y2 + 5.0f*x4 + y4);
		data_out[52] = 4.1513246297620823f*xy*z*(x2 - y2)*(13.0f*z2 - 3.0f);
		data_out[53] = -0.15645893386229404f*y*(3.0f*x2 - y2)*(13.0f*z2*(11.0f*z2 - 3.0f) - 27.0f*z2 + 3.0f);
		data_out[54] = 0.44253269244498261f*xy*z*(-110.0f*z2 + 143.0f*z4 + 15.0f);
		data_out[55] = 0.090331607582517306f*y*(-135.0f*z2 + 495.0f*z4 - 429.0f*z6 + 5.0f);
		data_out[56] = 0.068284276912004949f*z*(315.0f*z2 - 693.0f*z4 + 429.0f*z6 - 35.0f);
		data_out[57] = 0.090331607582517306f*x*(-135.0f*z2 + 495.0f*z4 - 429.0f*z6 + 5.0f);
		data_out[58] = 0.07375544874083044f*z*(x2 - y2)*(143.0f*z2*(3.0f*z2 - 1.0f) - 187.0f*z2 + 45.0f);
		data_out[59] = -0.15645893386229404f*x*(x2 - 3.0f*y2)*(13.0f*z2*(11.0f*z2 - 3.0f) - 27.0f*z2 + 3.0f);
		data_out[60] = 1.0378311574405206f*z*(13.0f*z2 - 3.0f)*(-6.0f*x2*y2 + x4 + y4);
		data_out[61] = -0.51891557872026028f*x*(13.0f*z2 - 1.0f)*(-10.0f*x2*y2 + x4 + 5.0f*y4);
		data_out[62] = 2.6459606618019f*z*(15.0f*x2*y4 - 15.0f*x4*y2 + x6 - y6);
		data_out[63] = 0.70716273252459627f*x*(-35.0f*x2*y4 + 21.0f*x4*y2 - x6 + 7.0f*y6);
	};

	fill_sh();
}

torch::Tensor CuSHEncoderImpl :: CuSHEncode(const torch::Tensor &input)
{
	CHECK(input.is_contiguous());
	int n_pts = input.size(0);
	torch::Tensor result = torch::empty({ n_pts, Degree * Degree }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).contiguous();
	dim3 grid_dim { unsigned(((n_pts) + (512u) - 1) / (512u)), 1, 1 }; 
	dim3 block_dim { 512u, 1, 1 };
	CuSHKernel<<<grid_dim, block_dim>>>(n_pts, Degree, input.data_ptr<float>(), result.data_ptr<float>());
	return result;
}