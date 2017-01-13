#ifndef NOISE_SIMPLEX_FUNC
#define NOISE_SIMPLEX_FUNC
/*

Description:
	Array- and textureless CgFx/HLSL 2D, 3D and 4D simplex noise functions.
	a.k.a. simplified and optimized Perlin noise.

	The functions have very good performance
	and no dependencies on external data.

	2D - Very fast, very compact code.
	3D - Fast, compact code.
	4D - Reasonably fast, reasonably compact code.

------------------------------------------------------------------

Ported by:
	Lex-DRL
	I've ported the code from GLSL to CgFx/HLSL for Unity,
	added a couple more optimisations (to speed it up even further)
	and slightly reformatted the code to make it more readable.

Original GLSL functions:
	https://github.com/ashima/webgl-noise
	Credits from original glsl file are at the end of this cginc.

------------------------------------------------------------------

Usage:

	float ns = snoise(v);
	// v is any of: float2, float3, float4

	Return type is float.
	To generate 2 or more components of noise (colorful noise),
	call these functions several times with different
	constant offsets for the arguments.
	E.g.:

	float3 colorNs = float3(
		snoise(v),
		snoise(v + 17.0),
		snoise(v - 43.0),
	);


Remark about those offsets from the original author:

	People have different opinions on whether these offsets should be integers
	for the classic noise functions to match the spacing of the zeroes,
	so we have left that for you to decide for yourself.
	For most applications, the exact offsets don't really matter as long
	as they are not too small or too close to the noise lattice period
	(289 in this implementation).

*/

// 1 / 289
#define NOISE_SIMPLEX_1_DIV_289 0.00346020761245674740484429065744f

float mod289(float x) {
	return x - floor(x * NOISE_SIMPLEX_1_DIV_289) * 289.0;
}

float2 mod289(float2 x) {
	return x - floor(x * NOISE_SIMPLEX_1_DIV_289) * 289.0;
}

float3 mod289(float3 x) {
	return x - floor(x * NOISE_SIMPLEX_1_DIV_289) * 289.0;
}

float4 mod289(float4 x) {
	return x - floor(x * NOISE_SIMPLEX_1_DIV_289) * 289.0;
}


// ( x*34.0 + 1.0 )*x =
// x*x*34.0 + x
float permute(float x) {
	return mod289(
		x*x*34.0 + x
	);
}

float3 permute(float3 x) {
	return mod289(
		x*x*34.0 + x
	);
}

float4 permute(float4 x) {
	return mod289(
		x*x*34.0 + x
	);
}



float4 grad4(float j, float4 ip)
{
	const float4 ones = float4(1.0, 1.0, 1.0, -1.0);
	float4 p, s;
	p.xyz = floor( frac(j * ip.xyz) * 7.0) * ip.z - 1.0;
	p.w = 1.5 - dot( abs(p.xyz), ones.xyz );

	// GLSL: lessThan(x, y) = x < y
	// HLSL: 1 - step(y, x) = x < y
	p.xyz -= sign(p.xyz) * (p.w < 0);

	return p;
}



// ----------------------------------- 2D -------------------------------------

float snoise(float2 v)
{
	const float4 C = float4(
		0.211324865405187, // (3.0-sqrt(3.0))/6.0
		0.366025403784439, // 0.5*(sqrt(3.0)-1.0)
	 -0.577350269189626, // -1.0 + 2.0 * C.x
		0.024390243902439  // 1.0 / 41.0
	);

// First corner
	float2 i = floor( v + dot(v, C.yy) );
	float2 x0 = v - i + dot(i, C.xx);

// Other corners
	// float2 i1 = (x0.x > x0.y) ? float2(1.0, 0.0) : float2(0.0, 1.0);
	// Lex-DRL: afaik, step() in GPU is faster than if(), so:
	// step(x, y) = x <= y

	// Actually, a simple conditional without branching is faster than that madness :)
	int2 i1 = (x0.x > x0.y) ? float2(1.0, 0.0) : float2(0.0, 1.0);
	float4 x12 = x0.xyxy + C.xxzz;
	x12.xy -= i1;

// Permutations
	i = mod289(i); // Avoid truncation effects in permutation
	float3 p = permute(
		permute(
				i.y + float3(0.0, i1.y, 1.0 )
		) + i.x + float3(0.0, i1.x, 1.0 )
	);

	float3 m = max(
		0.5 - float3(
			dot(x0, x0),
			dot(x12.xy, x12.xy),
			dot(x12.zw, x12.zw)
		),
		0.0
	);
	m = m*m ;
	m = m*m ;

// Gradients: 41 points uniformly over a line, mapped onto a diamond.
// The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

	float3 x = 2.0 * frac(p * C.www) - 1.0;
	float3 h = abs(x) - 0.5;
	float3 ox = floor(x + 0.5);
	float3 a0 = x - ox;

// Normalise gradients implicitly by scaling m
// Approximation of: m *= inversesqrt( a0*a0 + h*h );
	m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );

// Compute final noise value at P
	float3 g;
	g.x = a0.x * x0.x + h.x * x0.y;
	g.yz = a0.yz * x12.xz + h.yz * x12.yw;
	return 130.0 * dot(m, g);
}

// ----------------------------------- 3D -------------------------------------

float snoise(float3 v)
{
	const float2 C = float2(
		0.166666666666666667, // 1/6
		0.333333333333333333  // 1/3
	);
	const float4 D = float4(0.0, 0.5, 1.0, 2.0);

// First corner
	float3 i = floor( v + dot(v, C.yyy) );
	float3 x0 = v - i + dot(i, C.xxx);

// Other corners
	float3 g = step(x0.yzx, x0.xyz);
	float3 l = 1 - g;
	float3 i1 = min(g.xyz, l.zxy);
	float3 i2 = max(g.xyz, l.zxy);

	float3 x1 = x0 - i1 + C.xxx;
	float3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
	float3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

// Permutations
	i = mod289(i);
	float4 p = permute(
		permute(
			permute(
					i.z + float4(0.0, i1.z, i2.z, 1.0 )
			) + i.y + float4(0.0, i1.y, i2.y, 1.0 )
		) 	+ i.x + float4(0.0, i1.x, i2.x, 1.0 )
	);

// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
	float n_ = 0.142857142857; // 1/7
	float3 ns = n_ * D.wyz - D.xzx;

	float4 j = p - 49.0 * floor(p * ns.z * ns.z); // mod(p,7*7)

	float4 x_ = floor(j * ns.z);
	float4 y_ = floor(j - 7.0 * x_ ); // mod(j,N)

	float4 x = x_ *ns.x + ns.yyyy;
	float4 y = y_ *ns.x + ns.yyyy;
	float4 h = 1.0 - abs(x) - abs(y);

	float4 b0 = float4( x.xy, y.xy );
	float4 b1 = float4( x.zw, y.zw );

	//float4 s0 = float4(lessThan(b0,0.0))*2.0 - 1.0;
	//float4 s1 = float4(lessThan(b1,0.0))*2.0 - 1.0;
	float4 s0 = floor(b0)*2.0 + 1.0;
	float4 s1 = floor(b1)*2.0 + 1.0;
	float4 sh = -step(h, 0.0);

	float4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
	float4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

	float3 p0 = float3(a0.xy,h.x);
	float3 p1 = float3(a0.zw,h.y);
	float3 p2 = float3(a1.xy,h.z);
	float3 p3 = float3(a1.zw,h.w);

//Normalise gradients
	float4 norm = rsqrt(float4(
		dot(p0, p0),
		dot(p1, p1),
		dot(p2, p2),
		dot(p3, p3)
	));
	p0 *= norm.x;
	p1 *= norm.y;
	p2 *= norm.z;
	p3 *= norm.w;

// Mix final noise value
	float4 m = max(
		0.6 - float4(
			dot(x0, x0),
			dot(x1, x1),
			dot(x2, x2),
			dot(x3, x3)
		),
		0.0
	);
	m = m * m;
	return 42.0 * dot(
		m*m,
		float4(
			dot(p0, x0),
			dot(p1, x1),
			dot(p2, x2),
			dot(p3, x3)
		)
	);
}

// ----------------------------------- 4D -------------------------------------

float snoise(float4 v)
{
	const float4 C = float4(
		0.138196601125011, // (5 - sqrt(5))/20 G4
		0.276393202250021, // 2 * G4
		0.414589803375032, // 3 * G4
	 -0.447213595499958  // -1 + 4 * G4
	);

// First corner
	float4 i = floor(
		v +
		dot(
			v,
			0.309016994374947451 // (sqrt(5) - 1) / 4
		)
	);
	float4 x0 = v - i + dot(i, C.xxxx);

// Other corners

// Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
	float4 i0;
	float3 isX = step( x0.yzw, x0.xxx );
	float3 isYZ = step( x0.zww, x0.yyz );
	i0.x = isX.x + isX.y + isX.z;
	i0.yzw = 1.0 - isX;
	i0.y += isYZ.x + isYZ.y;
	i0.zw += 1.0 - isYZ.xy;
	i0.z += isYZ.z;
	i0.w += 1.0 - isYZ.z;

	// i0 now contains the unique values 0,1,2,3 in each channel
	float4 i3 = saturate(i0);
	float4 i2 = saturate(i0-1.0);
	float4 i1 = saturate(i0-2.0);

	//	x0 = x0 - 0.0 + 0.0 * C.xxxx
	//	x1 = x0 - i1  + 1.0 * C.xxxx
	//	x2 = x0 - i2  + 2.0 * C.xxxx
	//	x3 = x0 - i3  + 3.0 * C.xxxx
	//	x4 = x0 - 1.0 + 4.0 * C.xxxx
	float4 x1 = x0 - i1 + C.xxxx;
	float4 x2 = x0 - i2 + C.yyyy;
	float4 x3 = x0 - i3 + C.zzzz;
	float4 x4 = x0 + C.wwww;

// Permutations
	i = mod289(i);
	float j0 = permute(
		permute(
			permute(
				permute(i.w) + i.z
			) + i.y
		) + i.x
	);
	float4 j1 = permute(
		permute(
			permute(
				permute (
					i.w + float4(i1.w, i2.w, i3.w, 1.0 )
				) + i.z + float4(i1.z, i2.z, i3.z, 1.0 )
			) + i.y + float4(i1.y, i2.y, i3.y, 1.0 )
		) + i.x + float4(i1.x, i2.x, i3.x, 1.0 )
	);

// Gradients: 7x7x6 points over a cube, mapped onto a 4-cross polytope
// 7*7*6 = 294, which is close to the ring size 17*17 = 289.
	const float4 ip = float4(
		0.003401360544217687075, // 1/294
		0.020408163265306122449, // 1/49
		0.142857142857142857143, // 1/7
		0.0
	);

	float4 p0 = grad4(j0, ip);
	float4 p1 = grad4(j1.x, ip);
	float4 p2 = grad4(j1.y, ip);
	float4 p3 = grad4(j1.z, ip);
	float4 p4 = grad4(j1.w, ip);

// Normalise gradients
	float4 norm = rsqrt(float4(
		dot(p0, p0),
		dot(p1, p1),
		dot(p2, p2),
		dot(p3, p3)
	));
	p0 *= norm.x;
	p1 *= norm.y;
	p2 *= norm.z;
	p3 *= norm.w;
	p4 *= rsqrt( dot(p4, p4) );

// Mix contributions from the five corners
	float3 m0 = max(
		0.6 - float3(
			dot(x0, x0),
			dot(x1, x1),
			dot(x2, x2)
		),
		0.0
	);
	float2 m1 = max(
		0.6 - float2(
			dot(x3, x3),
			dot(x4, x4)
		),
		0.0
	);
	m0 = m0 * m0;
	m1 = m1 * m1;

	return 49.0 * (
		dot(
			m0*m0,
			float3(
				dot(p0, x0),
				dot(p1, x1),
				dot(p2, x2)
			)
		) + dot(
			m1*m1,
			float2(
				dot(p3, x3),
				dot(p4, x4)
			)
		)
	);
}



//                 Credits from source glsl file:
//
// Description : Array and textureless GLSL 2D/3D/4D simplex
//               noise functions.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : ijm
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//
//
//           The text from LICENSE file:
//
//
// Copyright (C) 2011 by Ashima Arts (Simplex noise)
// Copyright (C) 2011 by Stefan Gustavson (Classic noise)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#endif


// fractal sum
float fBm(float3 p, int octaves, float lacunarity = 2.0, float gain = 0.5)
{
	float freq = 1.0, amp = 1.0;
	float sum = 0;
	for(int i=0; i<octaves; i++) {
		sum += snoise(p*freq)*amp;
		freq *= lacunarity;
		amp *= gain;
	}
	return sum;
}

float turbulence(float3 p, int octaves, float lacunarity = 2.0, float gain = 0.5)// 0.5)
{
	float sum = 0;
	float freq = 1.0, amp = 1.0;
	for(int i=0; i<octaves; i++) {
		sum += abs(snoise(p*freq))*amp;
		freq *= lacunarity;
		amp *= gain;
	}
	return sum;
}

float turbulenceZ(float3 p, int octaves, float lacunarity = 2.0, float gain = 0.5)// 0.5)
{
	float sum = 0;
	float freq = 1.0, amp = 1.0;
	for(int i=0; i<octaves; i++) {
		sum += (snoise(p*freq))*amp;
		freq *= lacunarity;
		amp *= gain;
	}
	return min(max(sum, -1.0), 1.0);
}

// Ridged multifractal
// See "Texturing & Modeling, A Procedural Approach", Chapter 12
float ridge(float h, float offset)
{
    h = abs(h);
    h = offset - h;
    h = h * h;
    return h;
}

float ridgedmf(float3 p, int octaves, float lacunarity = 2.0, float gain = 0.5, float offset = 1.0)
{
	// Hmmm... these hardcoded constants make it look nice.  Put on tweakable sliders?
	float f = 0.3 + 0.5 * fBm(p, octaves, lacunarity, gain);
	return ridge(f, offset);
}

// mixture of ridged and fbm noise
float hybridTerrain(float3 x, int3 octaves)
{
	const float SCALE = 256;
	x /= SCALE;

	const int RIDGE_OCTAVES = octaves.x;
	const int FBM_OCTAVES   = octaves.y;
	const int TWIST_OCTAVES = octaves.z;
	const float LACUNARITY = 2, GAIN = 0.5;

	// Distort the ridge texture coords.  Otherwise, you see obvious texel edges.
	//float2 xOffset = float3(fBm(0.2*x, TWIST_OCTAVES), fBm(0.2*x+0.2, TWIST_OCTAVES));
	float3 xTwisted = x + 0.01 ;//* xOffset;

	// Ridged is too ridgy.  So interpolate between ridge and fBm for the coarse octaves.
	float h = ridgedmf(xTwisted, RIDGE_OCTAVES, LACUNARITY, GAIN, 1.0);

	const float fBm_UVScale  = pow(LACUNARITY, RIDGE_OCTAVES);
	const float fBm_AmpScale = pow(GAIN,       RIDGE_OCTAVES);
	float f = fBm(x * fBm_UVScale, FBM_OCTAVES, LACUNARITY, GAIN) * fBm_AmpScale;

	if (RIDGE_OCTAVES > 0)
		return h + f*saturate(h);
	else
		return f;
}


// This allows us to compile the shader with a #define to choose
// the different partition modes for the hull shader.
// See the hull shader: [partitioning(BEZIER_HS_PARTITION)]
// This sample demonstrates "integer", "fractional_even", and "fractional_odd"
#ifndef HS_PARTITION
#define HS_PARTITION "integer"
#endif //HS_PARTITION

// The input patch size.  In this sample, it is 3 vertices(control points).
// This value should match the call to IASetPrimitiveTopology()
#define INPUT_PATCH_SIZE 4

// The output patch size.  In this sample, it is also 3 vertices(control points).
#define OUTPUT_PATCH_SIZE 4


//--------------------------------------------------------------------------------------
// Vertex shader section
//--------------------------------------------------------------------------------------
struct VS_CONTROL_POINT_INPUT
{
    float4 vPosition        : SV_Position;
		float3 color : COLOR;

};

struct VS_CONTROL_POINT_OUTPUT
{
    float4 vPosition        : SV_Position;
		float3 pos : POSITION;

		float3 color : COLOR;
};

//----

cbuffer Locals {
	float4x4 u_Model;
	float4x4 u_View;
	float4x4 u_Proj;
	float4 u_ipos;
	float4 u_cpos;
	float4 u_color;
	float u_iscale;
};

struct VsOutput {
    float4 pos: SV_Position;
    float3 color: COLOR;
		float3 prepos: WORLDPOS;
};


float4x4 build_transform(float3 pos, float3 ang)
{
  float cosX = cos(ang.x);
  float sinX = sin(ang.x);
  float cosY = cos(ang.y);
  float sinY = sin(ang.y);
  float cosZ = cos(ang.z);
  float sinZ = sin(ang.z);

  float4x4 m;

  float m00 = cosY * cosZ + sinX * sinY * sinZ;
  float m01 = cosY * sinZ - sinX * sinY * cosZ;
  float m02 = cosX * sinY;
  float m03 = 0.0;

  float m04 = -cosX * sinZ;
  float m05 = cosX * cosZ;
  float m06 = sinX;
  float m07 = 0.0;

  float m08 = sinX * cosY * sinZ - sinY * cosZ;
  float m09 = -sinY * sinZ - sinX * cosY * cosZ;
  float m10 = cosX * cosY;
  float m11 = 0.0;

  float m12 = pos.x;
  float m13 = pos.y;
  float m14 = pos.z;
  float m15 = 1.0;

  /*
  //------ Orientation ---------------------------------
  m[0] = vec4(m00, m01, m02, m03); // first column.
  m[1] = vec4(m04, m05, m06, m07); // second column.
  m[2] = vec4(m08, m09, m10, m11); // third column.

  //------ Position ------------------------------------
  m[3] = vec4(m12, m13, m14, m15); // fourth column.
  */

  //------ Orientation ---------------------------------
  m[0][0] = m00; // first entry of the first column.
  m[0][1] = m01; // second entry of the first column.
  m[0][2] = m02;
  m[0][3] = m03;

  m[1][0] = m04; // first entry of the second column.
  m[1][1] = m05; // second entry of the second column.
  m[1][2] = m06;
  m[1][3] = m07;

  m[2][0] = m08; // first entry of the third column.
  m[2][1] = m09; // second entry of the third column.
  m[2][2] = m10;
  m[2][3] = m11;

  //------ Position ------------------------------------
  m[3][0] = m12; // first entry of the fourth column.
  m[3][1] = m13; // second entry of the fourth column.
  m[3][2] = m14;
  m[3][3] = m15;

  return m;
}




VS_CONTROL_POINT_OUTPUT Vertex(float3 pos : a_Pos, float3 color : a_Color) {
	/*
float local_scale =(abs(-5000.0f+length(u_cpos))/7000.0f)+0.15;
		//float local_scale = lerp(0.1, 5.5,((-5000.0f+length(u_cpos))/5000.0f));

		local_scale = min(local_scale, 1.0);
		local_scale = max(local_scale, 0.001);
		if (local_scale > 0.6)
			local_scale = 1.0;*/
    //float3 preposa = u_ipos+(pos);

//		float3 prepos = u_ipos+(pos*u_iscale*float3(local_scale,local_scale,1.0f));
    float3 prepos = u_ipos+(pos*u_iscale);
    float minlength = 25.0/256.0;
	//	float nscale = 0.009;
	float nscale = 0.002;

    //prepos.z = snoise(prepos.xy*0.05);
    //prepos.z = snoise(prepos.xy*0.05);
	//	float theight =  snoise(prepos.xy*nscale);



    float md = pos.x+pos.y;

    //prepos.z = prepos.z * 5.0;




/*
			    if (u_iscale < 1.0 && (pos.x >= 25.0 || pos.x <= -25.0))

			        float b = prepos.y/(minlength*u_iscale);
			        float a = floor(b*u_iscale);
			        float c = ceil(b*u_iscale);
			        float a1 = a*minlength;
			        float c1 = c*minlength;
			        float nb =snoise(float2(prepos.x,a1).xy*nscale);
			        float na =snoise(float2(prepos.x, c1).xy*nscale);
			        //prepos.z = (nb+na)/2.0;
							theight = (nb+na)/2.0;
			        color = float3(1.0, 0.0, 0.0);
			    }*/



		float radi = 25.0*100.0*2.0;
		float3 cofg = float3(0.0,0.0,-radi);
		float3 spherevec = normalize((prepos)-cofg);
		float3 prepos3 =  ( radi*spherevec)+cofg;


		//prepos3 = prepos+float3(0.0, 0.0,theight*100.0);
	prepos3 = prepos3 - u_ipos;
	//float4 inner = mul(u_Model, float4(prepos3, 1.0));
	float3 inner = mul(u_Model, float4(prepos3, 1.0));

	float ninner = inner.xyz;
		//float theight2 =  hybridTerrain(inner.xyz*nscale/20.0);
		float theight =  1.0;//hybridTerrain(inner.xyz*nscale*100.0, 64.0);


			if (theight > 0.4)
				color = lerp(float3(0.2, 0.7, 0.2), float3(0.9, 0.9, 0.9), (theight-0.8)/0.2); // white
			else if (theight > 0.2)
				color = float3(0.2, 0.7, 0.2); // green
			else
					color = float3(0.2, 0.2, 0.7); // blue*/
			//inner = inner + (float4(spherevec, 1.0)*theight*200.0);
			inner = (normalize(inner) * radi) + (normalize(inner) *theight*50.0);
				// IT WORKS float4 p = mul(u_Proj, mul(u_View, float4(inner, 1.0)));
				float4 p = float4(inner, 1.0);
		//color = u_color;
    VS_CONTROL_POINT_OUTPUT output;
		output.vPosition=  p;
		output.pos = inner;
		///output.color = color;
		//output.prepos = inner;
    return output;
}

//--------------------------------------------------------------------------------------
// Evaluation domain shader section
//--------------------------------------------------------------------------------------
struct DS_OUTPUT
{
    float4 vPosition        : SV_Position;
		float3 vPosition2 : WORLDPOS;
		float3 vNormal : POSITION;
		float3 color : COLOR;
};


float SphereIntersect(float3 ro, float3 ray)
{
  float3 v = float3(0,0,0) - ro;
  float b = dot(v, ray);
  float disc = (b*b) - dot(v, v) + 5000.0;

//  hit = true;
  // note - early returns not supported by HLSL compiler currently:
//  if (disc<=0) return -1.0; // ray misses
  //if (disc<=0) hit = false;;

  disc = sqrt(disc);
  float t2 = b + disc;

//  if (t2<=eps) return -1.0; // behind ray origin
//  if (t2<=eps) hit = false; // behind ray origin

  float t1 = b - disc;

  if ((t1>0.00001) && (t1<t2))  // return nearest intersection
    return t1;
  else
    return t2;
}
/*
#define AA 3


#define NB_ITER 64
#define PI 3.14159265359
#define TAO 6.28318531
#define MAX_DIST 4000.
#define PRECISION .0001



float Kaleido(inout vec2 v, in float nb){
	float id=floor(.5+ATAN(v.x,-v.y)*nb/TAO);
	float a = id*TAO/nb;
	float ca = COS(a), sa = SIN(a);
	v*=mat2(ca,sa,-sa,ca);
	return id;
}

vec2 Kaleido2(inout vec3 p, in float nb1, in float nb2, in float d) {
	float id1 = Kaleido(p.yx, nb1);
	float id2 = Kaleido(p.xz, nb2*2.);
	p.z+=d;
	return vec2(id1,id2);
}

float DECrater(float3 p) {
	float d = MAX_DIST;
	float2 id = Kaleido2(p, 9.,6.,2.);
	float noise1 = noise(id*10.);
	if (noise1<.6 && abs(id.y)>0.&&abs(id.y)<6.-1.) {
		d = sdCapsule(p, float3(0,0,-.15), float3(0,0,.1),.1+noise1*.2,.1+noise1*.5);
		d = max(-(length(p-float3(0,0,-.25))-(.1+noise1*.5)),d);
		d = max(-(length(p-float3(0,0,-.05))-(.1+noise1*.15)),d);
		d*=.8;
	}
	return d;
}

float4 DE(float3 p0) {
	float scalePlanet = 10.,
		  scaleFlag = 2.,
		  scaleAlien = .5;
	float4 res = float4(1000,1000,1000,1000);
	float3 p = p0;
    float d,d1,dx;
float AnimStep = 1.0;
//    if (withPlanet) {
	p = p0;
	p.x+=2.;
	p*=scalePlanet;
	bool withPlanet = true;
	//p.yz *= Rot1;
	//p.xz *= mat2(C2,S2,-S2,C2);
    if (withPlanet) {
	d1 = DECrater(p);
// Much better but cannot be render with the full scene in my computer
//	p.xz *= Rot1;
//	p.xy *= Rot1;
//	float d2 = DECrater(p);
	d = smin(length(p)-2.,d1,.15); //smin(d2, d1,.2),.15);

	d += .1*noise((p)*2.);
	d += .005*noise((p)*20.);
	res = float4(d/=scalePlanet,PLANET, length(p), p.z);

    }

	if (AnimStep >= 4) {
		dx = abs(C1);
		dx = clamp(dx,.1,.8);

		if (AnimStep == 4) {
			p = p0;
			p.x += (2.5*dx/scaleAlien) - 2.1;
		} else {
			p /= scalePlanet;
			//p.x-=1.;
		}
		p = p*scaleFlag;
		vec4 dFlag = DEFlag(p);
		dFlag.x /= scaleFlag;
		res = (dFlag.x<res.x) ? dFlag: res;
	}

	if (AnimStep > 1 && AnimStep < 7) {
		p = p0;
		if (AnimStep < 3) {
			p.x -= 3.2-steps(10.*(.038+time-5.25),.75);
			p.z -= 2.*
			(5.*pyramid(10.*(.038+time-5.25)));
		} else if(AnimStep>5) {
			p.x -= 3.2-steps(10.*(.038+6.75-time),.75);
			p.z -= 2.* (5.*pyramid(10.*(.038+6.75-time)));
		} else {
			p.x-=3.2;
		}
		p*=scaleAlien;
		vec4 dAlien = DEAlien(p);
		dAlien.x/=scaleAlien;
		res = (dAlien.x<res.x) ? dAlien: res;
	}
	return res;
}
*/

int simplexSeed;

int hash(int a)
{
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * simplexSeed;
    a = a ^ (a >> 15);
    return a;
}

/*
** 1D
*/

void grad1( int hash,  out float gx )
{
    int h = hash & 15;
    gx = 1.0f + (h & 7);
    if (h&8) gx = -gx;
}

float noise( float x )
{
  int i0 = floor(x);
  int i1 = i0 + 1;
  float x0 = x - i0;
  float x1 = x0 - 1.0f;

  float gx0, gx1;
  float n0, n1;
  float t20, t40, t21, t41;

  float x20 = x0 * x0;
  float t0 = 1.0f - x20;
  t20 = t0 * t0;
  t40 = t20 * t20;
  grad1(hash(i0 & 0xff), gx0);
  n0 = t40 * gx0 * x0;

  float x21 = x1*x1;
  float t1 = 1.0f - x21;
  t21 = t1 * t1;
  t41 = t21 * t21;
  grad1(hash(i1 & 0xff), gx1);
  n1 = t41 * gx1 * x1;

  return 0.25f * (n0 + n1);
}

float noise (float x, out float dnoise_dx)
{
  int i0 = floor(x);
  int i1 = i0 + 1;
  float x0 = x - i0;
  float x1 = x0 - 1.0f;

  float gx0, gx1;
  float n0, n1;
  float t20, t40, t21, t41;

  float x20 = x0 * x0;
  float t0 = 1.0f - x20;
  t20 = t0 * t0;
  t40 = t20 * t20;
  grad1(hash(i0 & 0xff), gx0);
  n0 = t40 * gx0 * x0;

  float x21 = x1*x1;
  float t1 = 1.0f - x21;
  t21 = t1 * t1;
  t41 = t21 * t21;
  grad1(hash(i1 & 0xff), gx1);
  n1 = t41 * gx1 * x1;

  dnoise_dx = t20 * t0 * gx0 * x20;
  dnoise_dx += t21 * t1 * gx1 * x21;
  dnoise_dx *= -8.0f;
  dnoise_dx += t40 * gx0 + t41 * gx1;
  dnoise_dx *= 0.1;
  return 0.25f * (n0 + n1);
}

/*
** 2D
*/

static    float2 grad2lut[8] =
   {
     { -1.0f, -1.0f }, { 1.0f, 0.0f } , { -1.0f, 0.0f } , { 1.0f, 1.0f } ,
     { -1.0f, 1.0f } , { 0.0f, -1.0f } , { 0.0f, 1.0f } , { 1.0f, -1.0f }
   };

float2 grad2( int hash)
{
    return grad2lut[hash & 7];
}

float noise( float2 input)
  {
    float n0, n1, n2;
    float2 g0, g1, g2;

    float s = ( input.x + input.y ) * 0.366025403f;
    float2 a = input + s;
    int2 ij = floor( a );

    float t = ( float ) ( ij.x + ij.y ) * 0.211324865f;
    float2 b = ij - t;
    float2 c = input - b;

    int2 ij1 = c.x > c.y ? float2(1,0) : float2(0,1);

   float2 c1 = c - ij1 + 0.211324865f;
   float2 c2 = c - 1.0f + 2.0f * 0.211324865f;

    int ii = ij.x & 0xff;
    int jj = ij.y & 0xff;

    float t0 = 0.5f - c.x * c.x - c.y * c.y;
    float t20, t40;
    if( t0 < 0.0f ) t40 = t20 = t0 = n0 = g0.x = g0.y = 0.0f;
    else
    {
      g0 = grad2( hash(ii + hash(jj)));
      t20 = t0 * t0;
      t40 = t20 * t20;
      n0 = t40 * ( g0.x * c.x + g0.y * c.y );
    }

    float t1 = 0.5f - c1.x * c1.x - c1.y * c1.y;
    float t21, t41;
    if( t1 < 0.0f ) t21 = t41 = t1 = n1 = g1.x = g1.y = 0.0f;
    else
    {
      g1 = grad2( hash(ii + ij1.x + hash(jj + ij1.y)));
      t21 = t1 * t1;
      t41 = t21 * t21;
      n1 = t41 * ( g1.x * c1.x + g1.y * c1.y );
    }

    float t2 = 0.5f - c2.x * c2.x - c2.y * c2.y;
    float t22, t42;
    if( t2 < 0.0f ) t42 = t22 = t2 = n2 = g2.x = g2.y = 0.0f;
    else
    {
      g2 = grad2( hash(ii + 1 + hash(jj + 1)));
      t22 = t2 * t2;
      t42 = t22 * t22;
      n2 = t42 * ( g2.x * c2.x + g2.y * c2.y );
    }

    float noise = 40.0f * ( n0 + n1 + n2 );
    return noise;
  }

float noise( float2 input, out float2 derivative)
  {
    float n0, n1, n2;
    float2 g0, g1, g2;

    float s = ( input.x + input.y ) * 0.366025403f;
    float2 a = input + s;
    int2 ij = floor( a );

    float t = ( float ) ( ij.x + ij.y ) * 0.211324865f;
    float2 b = ij - t;
    float2 c = input - b;

    int2 ij1 = c.x > c.y ? float2(1,0) : float2(0,1);

   float2 c1 = c - ij1 + 0.211324865f;
   float2 c2 = c - 1.0f + 2.0f * 0.211324865f;

    int ii = ij.x & 0xff;
    int jj = ij.y & 0xff;

    float t0 = 0.5f - c.x * c.x - c.y * c.y;
    float t20, t40;
    if( t0 < 0.0f ) t40 = t20 = t0 = n0 = g0.x = g0.y = 0.0f;
    else
    {
      g0 = grad2( hash(ii + hash(jj)));
      t20 = t0 * t0;
      t40 = t20 * t20;
      n0 = t40 * ( g0.x * c.x + g0.y * c.y );
    }

    float t1 = 0.5f - c1.x * c1.x - c1.y * c1.y;
    float t21, t41;
    if( t1 < 0.0f ) t21 = t41 = t1 = n1 = g1.x = g1.y = 0.0f;
    else
    {
      g1 = grad2( hash(ii + ij1.x + hash(jj + ij1.y)));
      t21 = t1 * t1;
      t41 = t21 * t21;
      n1 = t41 * ( g1.x * c1.x + g1.y * c1.y );
    }

    float t2 = 0.5f - c2.x * c2.x - c2.y * c2.y;
    float t22, t42;
    if( t2 < 0.0f ) t42 = t22 = t2 = n2 = g2.x = g2.y = 0.0f;
    else
    {
      g2 = grad2( hash(ii + 1 + hash(jj + 1)));
      t22 = t2 * t2;
      t42 = t22 * t22;
      n2 = t42 * ( g2.x * c2.x + g2.y * c2.y );
    }

    float noise = 40.0f * ( n0 + n1 + n2 );

    float temp0 = t20 * t0 * ( g0.x * c.x + g0.y * c.y );
    float temp1 = t21 * t1 * ( g1.x * c1.x + g1.y * c1.y );
    float temp2 = t22 * t2 * ( g2.x * c2.x + g2.y * c2.y );
    derivative = ((temp0 * c + temp1 * c1 + temp2 * c2) * -8 + (t40 * g0 + t41 * g1 + t42 * g2)) * 40;

    return noise;
  }

  /*
  ** 3D
  */

   static float3 grad3lut[16] =
   {
     { 1.0f, 0.0f, 1.0f }, { 0.0f, 1.0f, 1.0f },
     { -1.0f, 0.0f, 1.0f }, { 0.0f, -1.0f, 1.0f },
     { 1.0f, 0.0f, -1.0f }, { 0.0f, 1.0f, -1.0f },
     { -1.0f, 0.0f, -1.0f }, { 0.0f, -1.0f, -1.0f },
     { 1.0f, -1.0f, 0.0f }, { 1.0f, 1.0f, 0.0f },
     { -1.0f, 1.0f, 0.0f }, { -1.0f, -1.0f, 0.0f },
     { 1.0f, 0.0f, 1.0f }, { -1.0f, 0.0f, 1.0f },
     { 0.0f, 1.0f, -1.0f }, { 0.0f, -1.0f, -1.0f }
   };

   float3 grad3(int hash)
   {
      return grad3lut[hash & 15];
   }

  float simplexNoise(float3 input)
  {
    float n0, n1, n2, n3;
    float noise;
    float3 g0, g1, g2, g3;

    float s = (input.x + input.y + input.z) * 0.333333333;
    float3 a = input + s;
    int3 ijk = floor(a);

    float t = (float)(ijk.x + ijk.y + ijk.z) * 0.166666667;
    float3 b = ijk - t;
   float3 c = input - b;

   int3 ijk1;
   int3 ijk2;

    if(c.x >= c.y) {
      if(c.y >= c.z)
        { ijk1 = int3(1, 0, 0); ijk2 = int3(1,1,0); }
        else if(c.x >= c.z) { ijk1 = int3(1, 0, 0); ijk2 = int3(1,0,1); }
        else { ijk1 = int3(0, 0, 1); ijk2 = int3(1,0,1); }
      }
    else {
      if(c.y < c.z) { ijk1 = int3(0, 0, 1); ijk2 = int3(0,1,1); }
      else if(c.x < c.z) { ijk1 = int3(0, 1, 0); ijk2 = int3(0,1,1); }
      else { ijk1 = int3(0, 1, 0); ijk2 = int3(1,1,0); }
    }

    float3 c1 = c - ijk1 + 0.166666667;
   float3 c2 = c - ijk2 + 2.0f * 0.166666667;
   float3 c3 = c - 1.0f + 3.0f * 0.166666667;

    int ii = ijk.x & 0xff;
    int jj = ijk.y & 0xff;
    int kk = ijk.z & 0xff;

    float t0 = 0.6f - c.x * c.x - c.y * c.y - c.z * c.z;
    float t20, t40;
    if(t0 < 0.0f) n0 = t0 = t20 = t40 = g0.x = g0.y = g0.z = 0.0f;
    else {
      g0 = grad3( hash(ii + hash(jj + hash(kk))));
      t20 = t0 * t0;
      t40 = t20 * t20;
      n0 = t40 * ( g0.x * c.x + g0.y * c.y + g0.z * c.z );
    }

    float t1 = 0.6f - c1.x * c1.x -  c1.y * c1.y - c1.z * c1.z;
    float t21, t41;
    if(t1 < 0.0f) n1 = t1 = t21 = t41 = g1.x = g1.y = g1.z = 0.0f;
    else {
      g1 = grad3( hash(ii + ijk1.x + hash(jj + ijk1.y + hash(kk + ijk1.z))));
      t21 = t1 * t1;
      t41 = t21 * t21;
      n1 = t41 * ( g1.x * c1.x + g1.y * c1.y + g1.z * c1.z );
    }

    float t2 = 0.6f - c2.x * c2.x - c2.y * c2.y - c2.z * c2.z;
    float t22, t42;
    if(t2 < 0.0f) n2 = t2 = t22 = t42 = g2.x = g2.y = g2.z = 0.0f;
    else {
      g2 = grad3( hash(ii + ijk2.x + hash(jj + ijk2.y + hash(kk + ijk2.z))));
      t22 = t2 * t2;
      t42 = t22 * t22;
      n2 = t42 * ( g2.x * c2.x + g2.y * c2.y + g2.z * c2.z );
    }

    float t3 = 0.6f - c3.x * c3.x - c3.y * c3.y - c3.z * c3.z;
    float t23, t43;
    if(t3 < 0.0f) n3 = t3 = t23 = t43 = g3.x = g3.y = g3.z = 0.0f;
    else {
      g3 = grad3( hash(ii + 1 + hash(jj + 1 + hash(kk + 1))));
      t23 = t3 * t3;
      t43 = t23 * t23;
      n3 = t43 * ( g3.x * c3.x + g3.y * c3.y + g3.z * c3.z );
    }

    noise = 20.0f * (n0 + n1 + n2 + n3);
    return noise;
}

  float simplexNoise2( float3 input, out float3 derivative)
  {
    float n0, n1, n2, n3;
    float noise;
    float3 g0, g1, g2, g3;

    float s = (input.x + input.y + input.z) * 0.333333333;
    float3 a = input + s;
    int3 ijk = floor(a);

    float t = (float)(ijk.x + ijk.y + ijk.z) * 0.166666667;
    float3 b = ijk - t;
   float3 c = input - b;

   int3 ijk1;
   int3 ijk2;

    if(c.x >= c.y) {
      if(c.y >= c.z)
        { ijk1 = int3(1, 0, 0); ijk2 = int3(1,1,0); }
        else if(c.x >= c.z) { ijk1 = int3(1, 0, 0); ijk2 = int3(1,0,1); }
        else { ijk1 = int3(0, 0, 1); ijk2 = int3(1,0,1); }
      }
    else {
      if(c.y < c.z) { ijk1 = int3(0, 0, 1); ijk2 = int3(0,1,1); }
      else if(c.x < c.z) { ijk1 = int3(0, 1, 0); ijk2 = int3(0,1,1); }
      else { ijk1 = int3(0, 1, 0); ijk2 = int3(1,1,0); }
    }

    float3 c1 = c - ijk1 + 0.166666667;
   float3 c2 = c - ijk2 + 2.0f * 0.166666667;
   float3 c3 = c - 1.0f + 3.0f * 0.166666667;

    int ii = ijk.x & 0xff;
    int jj = ijk.y & 0xff;
    int kk = ijk.z & 0xff;

    float t0 = 0.6f - c.x * c.x - c.y * c.y - c.z * c.z;
    float t20, t40;
    if(t0 < 0.0f) n0 = t0 = t20 = t40 = g0.x = g0.y = g0.z = 0.0f;
    else {
      g0 = grad3( hash(ii + hash(jj + hash(kk))));
      t20 = t0 * t0;
      t40 = t20 * t20;
      n0 = t40 * ( g0.x * c.x + g0.y * c.y + g0.z * c.z );
    }

    float t1 = 0.6f - c1.x * c1.x -  c1.y * c1.y - c1.z * c1.z;
    float t21, t41;
    if(t1 < 0.0f) n1 = t1 = t21 = t41 = g1.x = g1.y = g1.z = 0.0f;
    else {
      g1 = grad3( hash(ii + ijk1.x + hash(jj + ijk1.y + hash(kk + ijk1.z))));
      t21 = t1 * t1;
      t41 = t21 * t21;
      n1 = t41 * ( g1.x * c1.x + g1.y * c1.y + g1.z * c1.z );
    }

    float t2 = 0.6f - c2.x * c2.x - c2.y * c2.y - c2.z * c2.z;
    float t22, t42;
    if(t2 < 0.0f) n2 = t2 = t22 = t42 = g2.x = g2.y = g2.z = 0.0f;
    else {
      g2 = grad3( hash(ii + ijk2.x + hash(jj + ijk2.y + hash(kk + ijk2.z))));
      t22 = t2 * t2;
      t42 = t22 * t22;
      n2 = t42 * ( g2.x * c2.x + g2.y * c2.y + g2.z * c2.z );
    }

    float t3 = 0.6f - c3.x * c3.x - c3.y * c3.y - c3.z * c3.z;
    float t23, t43;
    if(t3 < 0.0f) n3 = t3 = t23 = t43 = g3.x = g3.y = g3.z = 0.0f;
    else {
      g3 = grad3( hash(ii + 1 + hash(jj + 1 + hash(kk + 1))));
      t23 = t3 * t3;
      t43 = t23 * t23;
      n3 = t43 * ( g3.x * c3.x + g3.y * c3.y + g3.z * c3.z );
    }

    noise = 20.0f * (n0 + n1 + n2 + n3);

    float temp0 = t20 * t0 * ( g0.x * c.x + g0.y * c.y + g0.z * c.z );
    derivative = temp0 * c;
    float temp1 = t21 * t1 * ( g1.x * c1.x + g1.y * c1.y + g1.z * c1.z );
    derivative += temp1 * c1;
    float temp2 = t22 * t2 * ( g2.x * c2.x + g2.y * c2.y + g2.z * c2.z );
    derivative += temp2 * c2;
    float temp3 = t23 * t3 * ( g3.x * c3.x + g3.y * c3.y + g3.z * c3.z );
    derivative += temp3 * c3;
    derivative *= -8.0f;
    derivative += t40 * g0 + t41 * g1 + t42 * g2 + t43 * g3;
    derivative *= 28.0f;

    return noise;
}

float3 mod289_ahmed(float3 x)
{
    return x - floor(x / 289.0) * 289.0;
}

float4 mod289_ahmed(float4 x)
{
    return x - floor(x / 289.0) * 289.0;
}

float4 permute_ahmed(float4 x)
{
    return mod289_ahmed((x * 34.0 + 1.0) * x);
}

float4 taylorInvSqrt(float4 r)
{
    return 1.79284291400159 - r * 0.85373472095314;
}

float3 snoise_grad(float3 v)
{
    const float2 C = float2(1.0 / 6.0, 1.0 / 3.0);

    // First corner
    float3 i  = floor(v + dot(v, C.yyy));
    float3 x0 = v   - i + dot(i, C.xxx);

    // Other corners
    float3 g = step(x0.yzx, x0.xyz);
    float3 l = 1.0 - g;
    float3 i1 = min(g.xyz, l.zxy);
    float3 i2 = max(g.xyz, l.zxy);

    // x1 = x0 - i1  + 1.0 * C.xxx;
    // x2 = x0 - i2  + 2.0 * C.xxx;
    // x3 = x0 - 1.0 + 3.0 * C.xxx;
    float3 x1 = x0 - i1 + C.xxx;
    float3 x2 = x0 - i2 + C.yyy;
    float3 x3 = x0 - 0.5;

    // Permutations
    i = mod289_ahmed(i); // Avoid truncation effects in permutation
    float4 p =
      permute_ahmed(permute_ahmed(permute_ahmed(i.z + float4(0.0, i1.z, i2.z, 1.0))
                            + i.y + float4(0.0, i1.y, i2.y, 1.0))
                            + i.x + float4(0.0, i1.x, i2.x, 1.0));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    float4 j = p - 49.0 * floor(p / 49.0);  // mod(p,7*7)

    float4 x_ = floor(j / 7.0);
    float4 y_ = floor(j - 7.0 * x_);  // mod(j,N)

    float4 x = (x_ * 2.0 + 0.5) / 7.0 - 1.0;
    float4 y = (y_ * 2.0 + 0.5) / 7.0 - 1.0;

    float4 h = 1.0 - abs(x) - abs(y);

    float4 b0 = float4(x.xy, y.xy);
    float4 b1 = float4(x.zw, y.zw);

    //float4 s0 = float4(lessThan(b0, 0.0)) * 2.0 - 1.0;
    //float4 s1 = float4(lessThan(b1, 0.0)) * 2.0 - 1.0;
    float4 s0 = floor(b0) * 2.0 + 1.0;
    float4 s1 = floor(b1) * 2.0 + 1.0;
    float4 sh = -step(h, 0.0);

    float4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    float4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

    float3 g0 = float3(a0.xy, h.x);
    float3 g1 = float3(a0.zw, h.y);
    float3 g2 = float3(a1.xy, h.z);
    float3 g3 = float3(a1.zw, h.w);

    // Normalise gradients
    float4 norm = taylorInvSqrt(float4(dot(g0, g0), dot(g1, g1), dot(g2, g2), dot(g3, g3)));
    g0 *= norm.x;
    g1 *= norm.y;
    g2 *= norm.z;
    g3 *= norm.w;

    // Compute gradient of noise function at P
    float4 m = max(0.6 - float4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    float4 m2 = m * m;
    float4 m3 = m2 * m;
    float4 m4 = m2 * m2;
    float3 grad =
      -6.0 * m3.x * x0 * dot(x0, g0) + m4.x * g0 +
      -6.0 * m3.y * x1 * dot(x1, g1) + m4.y * g1 +
      -6.0 * m3.z * x2 * dot(x2, g2) + m4.z * g2 +
      -6.0 * m3.w * x3 * dot(x3, g3) + m4.w * g3;
    return 42.0 * grad;
}


float3 fBm3123(float3 p, int ioct)
{
	float x=p.x;
	float y=p.y;
	float z=p.z;

	float3 f = float3(0.0,0.0,0.0);
  float w = 0.5;
  float dx = 0.0;
  float dz = 0.0;
  for( int i=0; i < ioct ; i++ )
  {
      float3 n = snoise_grad( float3(x,y,z) );
      f.x += n.x;
      f.z += n.z;
      f.y += w * n.x / (1.0f + dx*dx + dz*dz); // replace with "w * n[0]" for a classic fbm()
w *= 0.5;
x *= 2.0;
y *= 2.0;
  }

	return f;
}

float3 fBm3(float3 p, int octaves, float lacunarity = 2.0, float gain = 0.5)
{
	float freq = 1.0, amp = 1.0;
	float3 sum = 0;
	float w = 0.5;
	float dx = 0.0;
	float dz = 0.0;
	for(int i=0; i<octaves; i++) {
		sum += snoise_grad(p*freq)*amp;
		freq *= lacunarity;
		amp *= gain;
	}
	return sum;
}
float3 CalculateSurfaceGradient(float3 n, float3 dpdx, float3 dpdy, float dhdx, float dhdy)
{
    float3 r1 = cross(dpdy, n);
    float3 r2 = cross(n, dpdx);

    return 1.0*(r1 * dhdx + r2 * dhdy) / dot(dpdx, r1);
}
// Move the normal away from the surface normal in the opposite surface gradient direction
float3 PerturbNormal(float3 n, float3 dpdx, float3 dpdy, float dhdx, float dhdy)
{
	 return normalize(n - CalculateSurfaceGradient(n, dpdx, dpdy, dhdx, dhdy));
}

float3 CalculateSurfaceNormal(float3 position, float3 normal, float height)
{
    float3 dpdx = ddx_fine(position);
    float3 dpdy = ddy_fine(position);

    float dhdx = ddx_fine(height);
    float dhdy = ddy_fine(height);

    return PerturbNormal(normal, dpdx, dpdy, dhdx, dhdy);
}

float ApplyChainRule(float dhdu, float dhdv, float dud_, float dvd_)
{
    return dhdu * dud_ + dhdv * dvd_;
}


float alerp(float a, float b, float c)
{
	return (c-a) / (b-a);
}

float value_hash(float3 p2)  // replace this by something better
{
	//if (abs(p2.x) < 0.00001)
	//	p2.x = 0.1;
    float3 p  = frac( p2*0.3183099+0.1+ float3(0.71,0.113,0.231) );
	p *= 17.0;

    return frac( p.x*p.y*p.z*(p.x+p.y+p.z) );
}

float value_noise(  float3 x )
{
    float3 p = floor(x);
    float3 f = frac(x);
    f = f*f*(3.0-2.0*f);

    return lerp(lerp(lerp( value_hash(p+float3(0,0,0)),
                        value_hash(p+float3(1,0,0)),f.x),
                   lerp( value_hash(p+float3(0,1,0)),
                        value_hash(p+float3(1,1,0)),f.x),f.y),
               lerp(lerp( value_hash(p+float3(0,0,1)),
                        value_hash(p+float3(1,0,1)),f.x),
                   lerp( value_hash(p+float3(0,1,1)),
                        value_hash(p+float3(1,1,1)),f.x),f.y),f.z);
}

float value_fBm(float3 p, int octaves, float lacunarity = 2.0, float gain = 0.5)
{
	gain = 0.33;
	float freq = 1.0, amp = 1.0;
	float sum = 0;
	for(int i=0; i<octaves; i++) {
		sum += value_noise(p*freq)*amp;
		freq *= lacunarity;
		amp *= gain;
	}

//	sum = 0.5 + 0.5*sum;

  //  sum *= smoothstep( 0.0, 0.005, abs(p.x-0.6) );

	return sum;
}

float value_hash2(float2 p)  // replace this by something better
{
    p  = 50.0*frac( p*0.3183099 + float2(0.71,0.113));
    return -1.0+2.0*frac( p.x*p.y*(p.x+p.y) );
}

float value_noise2( float2 p )
{
    float2 i = floor( p );
    float2 f = frac( p );

	float2 u = f*f*(3.0-2.0*f);

    return lerp( lerp( value_hash2( i + float2(0.0,0.0) ),
                     value_hash2( i + float2(1.0,0.0) ), u.x),
                lerp( value_hash2( i + float2(0.0,1.0) ),
                     value_hash2( i + float2(1.0,1.0) ), u.x), u.y);
}

float value_fBm2(float2 p, int octaves, float lacunarity = 2.0, float gain = 0.5)
{
	float freq = 1.0, amp = 1.0;
	float sum = 0;
	float2x2 m = float2x2( 1.6,  1.2, -1.2,  1.6 );
	for(int i=0; i<octaves; i++) {
		sum += value_noise2(p*freq)*amp;
		freq *= lacunarity;
		amp *= gain;
		p = mul(m,p);
	}

	sum = 0.5 + 0.5*sum;

    sum *= smoothstep( 0.0, 0.005, abs(p.x-0.6) );

//	sum = 0.5 + 0.5*sum;

  //  sum *= smoothstep( 0.0, 0.005, abs(p.x-0.6) );

	return sum;
}

float3x3 calcLookAt(float3 origin, float3 target, float roll) {
  float3 rr = float3(sin(roll), cos(roll), 0.0);
  float3 ww = normalize(target - origin);
  float3 uu = normalize(cross(ww, rr));
  float3 vv = normalize(cross(uu, ww));

  return float3x3(uu, vv, ww);
}

float3 myterrain(float3 p, float3 n, out float h)
{


		float nscale = 0.02;

	// float inz =h= value_fBm(p*nscale, 12);//*0.5+0.5;

	 float3 uvw=p*nscale;

	 float shift = 0.5;

/*
	 //float3 uvwB = mul(build_transform(uvw, float3(shift, 0.0, 0.0)),uvw) ;
	 //float3 uvwC = mul(build_transform(uvwB, float3(0.0, shift, 0.0)), uvwB);
	 const float mypi = 3.14159265358979323;
	 float sphere_u = 0.5 + (atan2(n.y, n.x) / (0.5 * mypi));
	 float sphere_v =  0.5 - (asin(n.z) / mypi);
*/


 //h= value_fBm2(float2(sphere_u*100.0, sphere_v*100.0), 12);;//value_fBm(p*nscale, 12);//*0.5+0.5;
 //h= //*0.5+0.5;



	//h*=h;
	//h = abs(1.0+res.y);;
	float3 kx = float3(1,0,0);
	float3 ky = float3(0,1,0);

/*
float hB = value_fBm2(float2((sphere_u*100.0)+shift, sphere_v*100.0), 12);;//
hB*=hB;
float hC = value_fBm2(float2(sphere_u*100.0, (sphere_v*100.0)+shift), 12);;//
hC*=hC;
*/
//mul(build_transform(float3(0.0,0.0,0.0), float3(shift, 0.0, 0.0)),uvw);

//float3 znx = float3(h - hC,  h - hB,1.0 ) ;//sqrt(pow(h - hB, 2) + pow(h - hC, 2) + 1);


h =value_fBm(uvw, 12);

		//res = normalize(res);

//for (int x=0; x<)
	//float3 bump = float3( value_fBm(pX, 12)*0.5+0.5,value_fBm(pY, 12)*0.5+0.5,value_fBm(pZ, 12)*0.5+0.5);
//float3 modNormal = float3( (bump.x-h) / E, (bump.y-h) / E, (bump.z-h) / E);

//float3 res = normalize(n - modNormal);

/*
 if (h< 0)
	{
	//	res.x = res.z = 0.0;
	//	res.y = 0;
		h *= 0.0012;
		res = lerp(n, res, 0.0012);
	}*/
	//res = n;
	//h = h*0.2;

	float3	res = CalculateSurfaceNormal(uvw, n, h);
	h *= h*h;
///	res = normalize(-mul(res, calcLookAt( float3(0,0,0), n, 0.0)));



	return res;
}
float3 fresnelSchlickRoughness(float cosTheta, float3 F0, float roughness)
{
    return F0 + (max(float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

float DistributionGGX(float3 N, float3 H, float roughness)
{
	const float PI = 3.14159265359;

    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}
float GeometrySmith(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

float4 Pixel(DS_OUTPUT pin) : SV_Target {


	//Interpolation to find each position the generated vertices
	float3 finalPos = pin.vPosition2;
	float nscaleO = 0.002;//0.002;
float nscale = 0.006;//0.002;
float shift = 0.01;



	float radi = 25.0*100.0*2.0;
	float3 cofg = float3(0.0,0.0,-radi);
	float3 spherevec = normalize((finalPos)-cofg);
	float3 prepos3 =  ( radi*spherevec)+cofg;

	float3 finalPosB = mul(build_transform(float3(0.0,0.0,0.0), float3(shift, 0.0, 0.0)),finalPos) ;
	float3 finalPosC = mul(build_transform(float3(0.0,0.0,0.0), float3(0.0, shift, 0.0)), finalPos);



	float4 finalPos2 = mul(u_Proj, mul(u_View, float4(finalPos, 1.0)));


	//float4 inner = mul(u_Model, float4(prepos3, 1.0));
	float3 inner = finalPos;



	//    Output.vPosition = mul( float4(finalPos,1), (u_View) );
	finalPos2 = mul(u_Proj, mul(u_View, float4(inner, 1.0)));
	float3 hnorm = normalize(finalPos);

	// IT WORKS float4 p = mul(u_Proj, mul(u_View, float4(inner, 1.0)));
float hzx = 0.0;
	float3 dn = myterrain(finalPos,hnorm, hzx);
	//float hzx = 1.0+(dn.y);
	float theightX = hzx;
	//float theightX =  snoise(finalPos*nscale);//, 32.0);//snoise(finalPos*nscaleO/9.0);//, 19.0);

		float theight = theightX;
	//theightX = 1.0 - theightX;
	/*
	float theight =  (theightX+turbulence(finalPos*nscale, 32.0));
		float theight1 =  (theightX+turbulence(finalPosB*nscale, 32.0));
			float theight2 =  (theightX+turbulence(finalPosC*nscale, 32.0));



			theight = theight*theight;
			theight1 = theight1*theight1;
				theight2 = theight2*theight2;
*/
		//	hnorm = mul(u_Model, hnorm);
//theight = pow(theight , 2.1);

// determine optimal number of layers


	///////////////////////////////////////////////////////////

	// previous texture coordinates



//return float4(0.1,0.8, 0.1, 1.0);
		float3 color = float3(244.0/256.0, 164.0/256.0, 96.0/256.0);
		if (theight > 0.85)
			color = float3(0.99, 0.99, 0.99); // white
			/*
		else if (theight > 0.9)
				color = float3(0.5, 0.2, 0.1); // greay
		else
		 if (theight > 0.5)
				color = float3(0.678431373, 0.360784314, 0.164705882); // greay
				*/
		else if (theight < 0.1)
		//	color = float3(0.2, 0.7, 0.2); // green
		//else
		{
				color = float3(0.2, 0.6, 0.2); // blue
				//	theight = 0.4;
			//		theight1 = 0.4;
			//		theight2 = 0.4;
		}
//color *= theight;
	//	float3 hnorm = normalize(float3(theight - theight1, 1.0, theight - theight2 ));
	float3 L = normalize(float3(1,0.1,1));

	//color = color* (theight);
	float initialHeight = 100.0;
	//if (theight < 0.4)
		//	color = float3(0.2, 0.2, 0.7); // blue
			float shadowMultiplier = 1;

			   const float minLayers = 15;
			   const float maxLayers = 30;
				 const float raylength = 200.0;

//color = float3(theight,theight,theight);
float3 ro = float3(-100000.0,0.0,0.0);
float3 rd = -L;
float mint = 0.02;
float maxt = 100.0;
float k = 7.0;

		// dn.y = abs(dn.y);
shadowMultiplier =max(0, dot(dn,hnorm))*1;
shadowMultiplier = shadowMultiplier;//*shadowMultiplier;
//color = color * max(0.02,max(0.0,dot(L,hnorm)) * max(0.5,max(0.0,dot(L,dn))) * shadowMultiplier);


//color = color * ((max(0.0,max(0.0,dot(L,hnorm)) *max(0.0,dot(L,dn))))) ;//+ (max(0.0,max(0.0,dot(-L,hnorm)) * max(shadowMultiplier,max(0.0,dot(-L,dn))))));
//color = color * ((max(0.2,lerp(max(0.0,dot(L,hnorm)) ,max(0.0,dot(L,dn)), 0.6)))) ;//+ (max(0.0,max(0.0,dot(-L,hnorm)) * max(shadowMultiplier,max(0.0,dot(-L,dn))))));
float metallic = 0.99f;
float roughness = 0.99;
float3 N = lerp(dn, hnorm, 0.0);
float3 WorldPos =  finalPos;
    float3 V = normalize(-u_cpos - WorldPos);

    float3 F0 = float3(0.04,0.04,0.04);
		float3 albedo = color;

    // reflectance equation
		float3 sunColor = float3(1,1,0.9);


    float3 Lo = float3(0.0,0.0,0.0);

	float diff = max(dot(N, L),  0.0);
	float3 diffuse =0.95*albedo* diff * sunColor;

float3 reflectDir = reflect(-L,N);
float3 spec = 0.0*pow(max(dot(normalize(V), reflectDir), 0.0), 5.0);
Lo = (diffuse+spec) ;
    float3 ambient = float3(135.0/256.0, 206.0/256.0, 250.0/256.0)*0.2 ;;//No ambient right now* ao;
    // color = (ambient) + Lo ;//+ float3(shadowMultiplier,shadowMultiplier,shadowMultiplier);
color = lerp((ambient + Lo)* shadowMultiplier, (ambient + Lo), 0.8) ;


//    color = color / (color + float3(1.0,1.0,1.0));
//    color = pow(color, 1.0/2.2);
/*
		float res = 1.0;
//		float h;
		float t = mint;
    for( int i=0; i<20; i++ )
		{
			float3 zzz = ro + (rd*t);
			float3 zzz2 = normalize(zzz);

			float hz1 = snoise(zzz );
			float hz2 = snoise( zzz2 );

			float mydot = dot(radi*zzz2+(hz2*100.0*normalize(zzz2)),normalize(zzz));
			 //float hz = snoise( normalize() );
		//	res = min( res, k*hz/t );
	    //            if( res<0.0001 ) break;
			if(mydot > 0) {res=0.0; break;}
			//t += 0.02;

    }
    shadowMultiplier =  clamp(res, 0., 1.);
*/

			   // calculate lighting only for surface oriented to the light source.
				 /*
			   if(dot(hnorm, L) > 0)
			   {
//					 shadowMultiplier = ;
			      // calculate initial parameters
			      float numSamplesUnderSurface	= 0.0;
			      float numLayers	= lerp(maxLayers, minLayers, abs(dot(hnorm, L)));
			      float layerHeight	= initialHeight / numLayers;
						float heightFromTexture	 = theight;
						for (float t=layerHeight; t<raylength; t+=layerHeight)
						{
							float ppp = SphereIntersect(float3(-10000,0,0), L);
							float3 ppp3 = (float3(-10000,0,0) + L*ppp)+(hnorm*500);
							float heightFromTexture2	= turbulence((radi*normalize(( (ppp3) + (-L*t)))*nscaleO), 32.0);//texture(u_heightTexture, currentTextureCoords).r;
							//shadowMultiplier = min(shadowMultiplier, heightFromTexture2);

							if (heightFromTexture2 > heightFromTexture)
							{
								shadowMultiplier *= 0.75;
								//break;
							}
							//else
							{
								//heightFromTexture = heightFromTexture2;
							}
						}
			      //vec2 texStep	= parallaxScale * L.xy / L.z / numLayers;

			      // current parameters
						/*
			      float currentLayerHeight	= initialHeight - layerHeight;
			      //float2 currentTextureCoords	= initialTexCoord + texStep;
			      float heightFromTexture	= theight;
			      int stepIndex	= 1;

			      // while point is below depth 0.0 )
			      while(currentLayerHeight > 0)
			      {
			         // if point is under the surface
			         if(heightFromTexture < currentLayerHeight)
			         {
			            // calculate partial shadowing factor
			            numSamplesUnderSurface	+= 1;
			            float newShadowMultiplier	= (currentLayerHeight - heightFromTexture) *
			                                             (1.0 - stepIndex / numLayers);
			            shadowMultiplier	= max(shadowMultiplier, newShadowMultiplier);
			         }

			         // offset to the next layer
			         stepIndex	+= 1;
			         currentLayerHeight	-= layerHeight;
			         //currentTextureCoords	+= texStep;
			         heightFromTexture	= snoise(radi*normalize( finalPos + (500.0*layerHeight*L*stepIndex/numLayers))*nscaleO/9.0);//texture(u_heightTexture, currentTextureCoords).r;
			      }

			      // Shadowing factor should be 1 if there were no points under the surface
			      if(numSamplesUnderSurface < 1)
			      {
			         shadowMultiplier = 1;
			      }
			      else
			      {
			         shadowMultiplier = 1.0 - shadowMultiplier;
			      }

			   }
				 else
				 {
					 	shadowMultiplier =0;
				 }*/

float3 hcolor = float3(theight,theight,theight);
	return float4(color, 1.0);
}


// This allows us to compile the shader with a #define to choose
// the different partition modes for the hull shader.
// See the hull shader: [partitioning(BEZIER_HS_PARTITION)]
// This sample demonstrates "integer", "fractional_even", and "fractional_odd"
#ifndef HS_PARTITION
#define HS_PARTITION "integer"
#endif //HS_PARTITION

// The input patch size.  In this sample, it is 3 vertices(control points).
// This value should match the call to IASetPrimitiveTopology()
#define INPUT_PATCH_SIZE 4

// The output patch size.  In this sample, it is also 3 vertices(control points).
#define OUTPUT_PATCH_SIZE 4

//----------------------------------------------------------------------------------
// Constant data function for the HS.  This is executed once per patch.
//--------------------------------------------------------------------------------------
struct HS_CONSTANT_DATA_OUTPUT
{
    float Edges[4]             : SV_TessFactor;
    float Inside [2]          : SV_InsideTessFactor;
};

struct HS_OUTPUT
{
    float3 vPosition           : POSITION;

};




float4 project(float3 pos) {

	float3 preposa = u_ipos+(pos);
	//float3 prepos = u_ipos+(pos*u_iscale);
	float3 prepos = u_ipos+(pos*u_iscale);
	float minlength = 25.0/256.0;
//	float nscale = 0.009;
float nscale = 0.002;



	float md = pos.x+pos.y;



	float radi = 25.0*100.0*2.0;
	float3 cofg = float3(0.0,0.0,-radi);
	float3 spherevec = normalize((prepos)-cofg);
	float3 prepos3 =  ( radi*spherevec)+cofg;


	//prepos3 = prepos+float3(0.0, 0.0,theight*100.0);
prepos3 = prepos3 - u_ipos;

	float4 result = mul(mul(mul(u_Model, prepos3), u_Proj), pos);
	result /= result.w;
	return result;
}

bool offscreen(float4 pos) {
	//return any(lessThan(pos.xy, vec2(-1.7)) || greaterThan(pos.xy, vec2(1.7)));
	if ((pos.x < -1000 || pos.y < -1000) || (pos.x > 1000.0 || pos.y >1000.0))
				return true;
	return false;
}

bool offscreen2(float3 pos) {
//	return true;
	//return any(lessThan(pos.xy, vec2(-1.7)) || greaterThan(pos.xy, vec2(1.7)));
//	return !(length(pos+u_cpos) < 500);

	float res = dot(normalize(pos+u_cpos), normalize(mul(u_View, float3(0,0,1))));
	return !(res < 0.3);
}


float level(float d, float camheight) {
	const float u_LODFactor = 0.15;//0.35f32,
	//return clamp((u_LODFactor+(0.25*(2.0-min(camheight/690.0, 2.0))))*740/d, 1, 64);
//	return clamp((u_LODFactor+(0.25*(2.0-min(camheight/690.0, 2.0))))*740/d, 1, 64);
}

float level2(float d, float3 campos, float3 vpos)
{
float radiiii = 5000.0f;
//float vv = max(0.0, min(alerp(0.0, d/(10.0),length(vpos+campos)), 1.0)) ;
float z = sqrt(d*d + (2.0*radiiii*d));
float vv = max(0.0, min(alerp(0.0, radiiii,z*(2.5)), 1.0)) ;
//float vv =(d/(5000.0*50.0));
//vv = min(1.0, vv)
//vv = vv*vv*vv*vv;
float finaltess = lerp(48, 2, vv);
return finaltess;
}
HS_CONSTANT_DATA_OUTPUT ConstantHS( InputPatch<VS_CONTROL_POINT_OUTPUT, 4> ip,
                                          uint PatchID : SV_PrimitiveID)
{



//	float3 campos = u_cpos.xyz;
//    HS_CONSTANT_DATA_OUTPUT Output;
HS_CONSTANT_DATA_OUTPUT Output;


	//	float4 v0 = project(ip[0].pos);
	//	float4 v1 = project(ip[1].pos);
	//	float4 v2 = project(ip[2].pos);
	//	float4 v3 = project(ip[3].pos);

	//	if (offscreen2(ip[0].pos) && offscreen2(ip[1].pos) && offscreen2(ip[2].pos) && offscreen2(ip[3].pos) )
		if (false)
		{
	//		Output.Edges[0] = Output.Edges[1] = Output.Edges[2] = Output.Edges[3] = 0;
	  //  Output.Inside [0] = Output.Inside [1] = 0;
		}
		else
		{
			/*
			float d0 = length(campos+ ip[0].pos.xyz);
			float d1 = length(campos+ ip[1].pos.xyz);
			float d2 = length(campos+ ip[2].pos.xyz);
			float d3 = length(campos+ip[3].pos.xyz);

			float avgd = (d0 + d1 + d2 + d3)/4.0;
            float h = length(campos);

			float3 vpos = (ip[0].pos+ip[1].pos+ip[2].pos+ip[3].pos)/4.0;
		  float vv = max(0.0, min(alerp(0.0, 5000/4.0,length(vpos+campos)), 1.0)) ;
			//vv = vv*vv*vv*vv;
			float finaltess = lerp(1, 60, (1.0-vv)*(1.0-vv)*(1.0-vv)*(1.0-vv));

			Output.Edges[0] = finaltess;//level2(avgd, campos, ip[0].pos.xyz);
			Output.Edges[1] =  finaltess;//level2(avgd, campos, ip[1].pos.xyz);
			Output.Edges[2] =  finaltess;//level2(avgd, campos, ip[2].pos.xyz);
			Output.Edges[3] =  finaltess;//level2(avgd, campos, ip[3].pos.xyz);
*/
	//		float l = max(max(Output.Edges[0], Output.Edges[1]), max(Output.Edges[2], Output.Edges[3]));
		//	Output.Inside[0] = finaltess;
			//Output.Inside[1] = finaltess;

			float max_tess = 64.0;
			float min_tess = 1.0;

		  float radiiii = 5000.0;
			//float3 campos = float3(u_View[3][0],u_View[3][1],u_View[3][2]);
			float3 campos = u_cpos.xyz;

			//float vv = max(0.0, min(alerp(radiiii, radiiii*2.0,length(campos)), 1.0)) ;
			float3 vpos = (ip[0].pos+ip[1].pos+ip[2].pos+ip[3].pos)/4.0;
			float vv = max(0.0, min(alerp(0.0, radiiii/4.0,length(vpos+campos)), 1.0)) ;
			//vv = vv*vv*vv*vv;
			float finaltess = lerp(min_tess, max_tess, (1.0-vv)*(1.0-vv)*(1.0-vv)*(1.0-vv));

			float g_fTessellationFactor = finaltess;//16.0;//8.0

		    Output.Edges[0] = Output.Edges[1] = Output.Edges[2] = Output.Edges[3] = g_fTessellationFactor;
		    Output.Inside [0] = Output.Inside [1] = g_fTessellationFactor;

		}

    //Output.Edges[0] = Output.Edges[1] = Output.Edges[2] = Output.Edges[3] = g_fTessellationFactor;
    //Output.Inside [0] = Output.Inside [1] = g_fTessellationFactor;

    return Output;
}

// The hull shader is called once per output control point, which is specified with
// outputcontrolpoints.  For this sample, we take the control points from the vertex
// shader and pass them directly off to the domain shader.  In a more complex scene,
// you might perform a basis conversion from the input control points into a Bezier
// patch, such as the SubD11 Sample of DirectX SDK.

// The input to the hull shader comes from the vertex shader

// The output from the hull shader will go to the domain shader.
// The tessellation factor, topology, and partition mode will go to the fixed function
// tessellator stage to calculate the UV and domain points.

[domain("quad")] //Quad domain for our shader
[partitioning(HS_PARTITION)] //Partitioning type according to the GUI
[outputtopology("triangle_cw")] //Where the generated triangles should face
[outputcontrolpoints(4)] //Number of times this part of the hull shader will be called for each patch
[patchconstantfunc("ConstantHS")] //The constant hull shader function
HS_OUTPUT HS( InputPatch<VS_CONTROL_POINT_OUTPUT, 4> p,
                    uint i : SV_OutputControlPointID,
                    uint PatchID : SV_PrimitiveID )
{
    HS_OUTPUT Output;
    Output.vPosition = p[i].vPosition;
/*
		Output.vPosition1 = p[0].vPosition;
		Output.vPosition2 = p[1].vPosition;
		Output.vPosition3 = p[2].vPosition;
		Output.vPosition4 = p[3].vPosition;*/
    return Output;
}


float3 swap_yz(float3 i)
{
		float y = i.y;
		return float3(i.x, i.z, y);
}
float3 cart2pol(float3 p)
{
    float r = sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
    float theta = atan2(p.y, p.x);
		float phi = atan2((sqrt(p.x*p.x + p.y*p.y)),p.z);
    return float3(r, theta, phi);
}
float3 pol2cart(float3 o)
{
		float3 z = float3(o.x * cos(o.y)* sin(o.z),
										  o.x * sin(o.y)*sin(o.z),
											o.x * cos(o.z));
    return z;
}

//--------------------------------------------------------------------------------------
struct PlanetMe
{
    float height;
		float3 color;


};


//Domain Shader is invoked for each vertex created by the Tessellator
[domain("quad")]
DS_OUTPUT DS( HS_CONSTANT_DATA_OUTPUT input,
                    float2 UV : SV_DomainLocation,
                    const OutputPatch<HS_OUTPUT, 4> quad )
{
  DS_OUTPUT Output;
		float nscaleO = 0.002;//0.002;
	float nscale = 0.006;//0.002;
	float shift = 0.01;
	//Interpolation to find each position the generated vertices
	float3 verticalPos1 = lerp(quad[0].vPosition,quad[1].vPosition,UV.y);
	float3 verticalPos2 = lerp(quad[3].vPosition,quad[2].vPosition,UV.y);
	float3 finalPos = lerp(verticalPos1,verticalPos2,UV.x);

	/*
	float3 finalPosPolarB = cart2pol(finalPos);
	finalPosPolarB.y += shift;
	float3 finalPosB = pol2cart(swap_yz(finalPosPolarB));

	float3 finalPosPolarC = cart2pol(finalPos);
	finalPosPolarC.z += shift;
	float3 finalPosC = pol2cart(swap_yz(finalPosPolarC));
	*/

float3 finalPosB = mul(build_transform(float3(0.0,0.0,0.0), float3(shift, 0.0, 0.0)),finalPos) ;
float3 finalPosC = mul(build_transform(float3(0.0,0.0,0.0), float3(0.0, shift, 0.0)), finalPos);



	float4 finalPos2 = mul(u_Proj, mul(u_View, float4(finalPos, 1.0)));




	float radi = 25.0*100.0*2.0;
	float3 cofg = float3(0.0,0.0,-radi);
	float3 spherevec = normalize((finalPos)-cofg);
	float3 prepos3 =  ( radi*spherevec)+cofg;


//float4 inner = mul(u_Model, float4(prepos3, 1.0));
float3 inner = finalPos;


			float theight;
			float3 nnn =  myterrain(finalPos, normalize(finalPos), theight);



	inner = (normalize(inner) * radi) + (normalize(inner) *theight*20.0);

finalPos2 = mul(u_Proj, mul(u_View, float4(inner, 1.0)));

Output.vPosition = finalPos2;
Output.vPosition2 = finalPos;

    return Output;
}
