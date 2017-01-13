// Terrain program
Texture2D<float4> t_BlitTex;
cbuffer Locals {
    float4x4 u_Model;
    float4x4 u_View;
    float4x4 u_Proj;
    float4 u_ipos;
		float4 u_cpos;
		float4 u_color;
    float u_iscale;
};
static const float3      iResolution = float3(1366.0, 768.0,0.0);
// math static const
static const float PI = 3.14159265359;
static const float DEG_TO_RAD = PI / 180.0;
static const float MAX = 10000.0;

// scatter static const
static const float K_R = 0.166;
static const float K_M = 0.0025;
static const float E = 14.3; 						// light intensity
static const float3  C_R = float3( 0.3, 0.7, 1.0 ); 	// 1 / wavelength ^ 4
static const float G_M = -0.85;					// Mie g

//static const float R = 1.0;
static const float R = 5500.0/1.0;
//static const float R_INNER = 0.7;
static const float R_INNER = 4900.0/1.0;
static const float SCALE_H = 4.0 / ( R - R_INNER );
static const float SCALE_L = 1.0 / ( R - R_INNER );

static const int NUM_OUT_SCATTER = 10;
static const float FNUM_OUT_SCATTER = 10.0;

static const int NUM_IN_SCATTER = 10;
static const float FNUM_IN_SCATTER = 10.0;

// angle : pitch, yaw
float3x3 rot3xy( float2 angle ) {
	float2 c = cos( angle );
	float2 s = sin( angle );

	return float3x3(
		c.y      ,  0.0, -s.y,
		s.y * s.x,  c.x,  c.y * s.x,
		s.y * c.x, -s.x,  c.y * c.x
	);
}

// ray direction
float3 ray_dir( float fov, float2 size, float2 pos ) {
	float2 xy = pos - size * 0.5;
	//xy.x /= 1366/768;

	float cot_half_fov = tan( ( 90.0 - fov * 0.5 ) * DEG_TO_RAD );
	float z = size.y * 0.5 * cot_half_fov;

	return normalize( float3( xy, -z ) );
}

// ray intersects sphere
// e = -b +/- sqrt( b^2 - c )
float2 ray_vs_sphere( float3 p, float3 dir, float r ) {
	float b = dot( p, dir );
	float c = dot( p, p ) - r * r;

	float d = b * b - c;
	if ( d < 0.0 ) {
		return float2( MAX, -MAX );
	}
	d = sqrt( d );

	return float2( -b - d, -b + d );
}

// Mie
// g : ( -0.75, -0.999 )
//      3 * ( 1 - g^2 )               1 + c^2
// F = ----------------- * -------------------------------
//      2 * ( 2 + g^2 )     ( 1 + g^2 - 2 * g * c )^(3/2)
float phase_mie( float g, float c, float cc ) {
	float gg = g * g;

	float a = ( 1.0 - gg ) * ( 1.0 + cc );

	float b = 1.0 + gg - 2.0 * g * c;
	b *= sqrt( b );
	b *= 2.0 + gg;

	return 1.5 * a / b;
}

// Reyleigh
// g : 0
// F = 3/4 * ( 1 + c^2 )
float phase_reyleigh( float cc ) {
	return 0.75 * ( 1.0 + cc );
}

float density( float3 p ){
	return exp( -( length( p ) - R_INNER ) * SCALE_H );
}

float optic( float3 p, float3 q ) {
	float3 step = ( q - p ) / FNUM_OUT_SCATTER;
	float3 v = p + step * 0.5;

	float sum = 0.0;
	for ( int i = 0; i < NUM_OUT_SCATTER; i++ ) {
		sum += density( v );
		v += step;
	}
	sum *= length( step ) * SCALE_L;

	return sum;
}

float3 in_scatter( float3 o, float3 dir, float2 e, float3 l ) {
	float len = ( e.y - e.x ) / FNUM_IN_SCATTER;
	float3 step = dir * len;
	float3 p = o + dir * e.x;
	float3 v = p + dir * ( len * 0.5 );

	float3 sum = float3( 0.0, 0.0 , 0.0  );
	for ( int i = 0; i < NUM_IN_SCATTER; i++ ) {
		float2 f = ray_vs_sphere( v, l, R );

		float3 u = v + l * f.y;

		float n = ( optic( p, v ) + optic( v, u ) ) * ( PI * 4.0 );

		sum += density( v ) * exp( -n * ( K_R * C_R + K_M ) );

		v += step;
	}
	sum *= len * SCALE_L;

	float c  = dot( dir, -l );
	float cc = c * c;

	return sum * ( K_R * C_R * phase_reyleigh( cc ) + K_M * phase_mie( G_M, c, cc ) ) * E;
}


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
float jSphere( float3 ro, float3 rd, float4 sph )
{
	float3 oc = ro - sph.xyz;
	float b = dot( oc, rd );
	float c = dot( oc, oc ) - sph.w*sph.w;
	float h = b*b - c;
	if( h<0.0 ) return -1.0;
	return -b - sqrt( h );
}

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


float fbm(float2 p)
{
	float z=0.5;
	float rz = 0.0;
	for (float i= 0.;i<3.;i++ )
	{
    rz = rz + (sin(snoise(p)*5.0)*0.5+0.5) *z;
		z *= 0.5;
		p = p*2.;
	}
	return rz;
}

float3 fog(float3 ro, float3 rd, float3 col, float ds)
{
	  float3 lgt = float3(-.523, .41, -.747);
    float3 pos = ro + rd*ds;
    float mx = (fbm(pos.zx*0.1-0.0*0.05)-0.5)*.2;

    const float b= 1.;
    float den = 0.3*exp(-ro.y*b)*(1.0-exp( -ds*rd.y*b ))/rd.y;
    float sdt = max(dot(rd, lgt), 0.);
    float3  fogColor  = lerp(float3(0.5,0.2,0.15)*1.2, float3(1.1,0.6,0.45)*1.3, pow(sdt,2.0)+mx*0.5);
    return lerp( col, fogColor, clamp(den + mx,0.,1.) );
}



#define iterations 17
#define formuparam 0.53

#define volsteps 20
#define stepsize 0.1

#define zoom   0.800
#define tile   0.850
#define speed  0.010

#define brightness 0.0015
#define darkmatter 0.300
#define distfading 0.730
#define saturation 0.850




float4 BlitPs(float4 pos: SV_Position): SV_Target {
	float4 o = t_BlitTex.Load(int3(pos.xy, 0));

// UNcomment below to disable render atmospheric shader
	return o; 

	float3 campos = -float3(u_View[3][0],u_View[3][1],u_View[3][2]);
	 //campos = float3(0.0 ,-9000.0,-1000.0);


	float2 sc = pos.xy /  iResolution;
														//	float3  rd =  normalize(mul(u_View,float3(sc.x, sc.y,1.0)));

															/*
				float			zzz = 								jSphere(campos, rd,float4(500.0,500.0,500.0,500.0));
				//if(zzz.x > 0.0)

				if ( zzz > 0.0 )
				{
					o = float4(1, 1.0, 1.0, 1.0);
				}

				return o;*/



				float3 dir = ray_dir( 60.0, iResolution.xy, float2(pos.x, pos.y) );

		// default ray origin
		float3 eye = float3( 0.0, 0.0, 0.0 );
		eye = u_cpos*1.0;//*2.0;

		// rotate camera
		float3x3 rot = rot3xy( float2( 0.0, 0.3 * 0.5 ) );

		dir = float3(dir.x, -dir.y,  dir.z);

		//dir = dir;
		eye = mul(u_View, eye);
		//eye = mul(u_View,mul(build_transform(eye, float3(0.0,0.0,0.0))));


		// sun light dir
		float3 l = normalize(float3( 1, 0, 1 ));
		l = normalize(mul(u_View, -l));

		float2 e = ray_vs_sphere( eye, dir, R );
		if ( e.x > e.y ) {
			//discard;

		}

		float2 f = ray_vs_sphere( eye, dir, R_INNER );




		e.y = min( e.y, f.x );

		float3 I = in_scatter( eye, dir, e, l );

		I.x = clamp(I.x, 0.0, 1.0);
		I.y = clamp(I.y, 0.0, 1.0);
		I.z = clamp(I.z, 0.0, 1.0);
		float Iw = I.x * I.y * I.z;

	///	if (length(u_ipos) < R)
	//		return o;

	float3 fogged = 		fog(eye, dir, o+I, 10.01);
	o =  o+float4(I, Iw);

	if ( f.x > f.y ) {
		//discard;
		float time=0.25;

		//mouse rotation

	float3 from= float3(1.,.5,0.5);
	from = mul(u_View, from);
		//volumetric rendering
		float s=0.1,fade=1.0;
		float3 v=float3(0,0,0);
		for (int r=0; r<volsteps; r++) {
			float3 p=from+s*dir*.5;
			p = abs(float3(tile,tile,tile)-fmod(p,float3(tile*2.,tile*2.,tile*2.))); // tiling fold
			float pa,a=pa=0.;
			for (int i=0; i<iterations; i++) {
				p=abs(p)/dot(p,p)-formuparam; // the magic formula
				a+=abs(length(p)-pa); // absolute sum of average change
				pa=length(p);
			}
			float dm=max(0.,darkmatter-a*a*.001); //dark matter
			a*=a*a; // add contrast
			if (r>6) fade*=1.-dm; // dark matter, don't render near
			//v+=vec3(dm,dm*.5,0.);
			v+=fade;
			v+=float3(s,s*s,s*s*s*s)*a*brightness*fade; // coloring based on distance
			fade*=distfading; // distance fading
			s+=stepsize;
		}
		float fl2 = length(v);
		v=lerp(float3(fl2,fl2,fl2),v,saturation); //color adjust
		float4 fragColor = float4(v*.01,1.);
		o = o + fragColor;
		//return o;
	}


	return o;
}

// common parts

float4 BlitVs(int2 pos: a_Pos): SV_Position {
	return float4(pos, 0.0, 1.0);
}
