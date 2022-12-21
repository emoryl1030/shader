// Author:CMH
// Title: Basic Raymarching_9(Hatching) 
// Reference: 20220414_glsl Breathing Circle_v5A(BRDF).qtz
#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;

#define PI 3.141592654
#define TWOPI 6.283185308
vec3 mapLadyBug( vec3 p );
float fbm(in vec2 uv);
float noise_3(in vec3 p) ;
vec3 normalMap(vec3 p, vec3 n);
vec3 hatching(vec3 p, vec3 n, float value);
vec3 FlameColour(float f);


//=== distance functions ===
float sdSphere( vec3 p, float s )
{
    return length(p)-s;
}
float sdBox( vec3 p, vec3 b )
{
  vec3 d = abs(p) - b;
  return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}
float sdTorus( vec3 p, vec2 t )
{
  vec2 q = vec2(length(p.xy)-t.x,p.z);
  return length(q)-t.y;
}

float map(in vec3 p)
{
float noise =0.03 * (noise_3(p*2.5)*2.0-1.0);
//return sdSphere(p+vec3(0.,0.,0.0), 0.5);
return sdTorus(p+vec3(0.,0.,0.0),vec2(0.4,0.2));
//return sdBox(p+vec3(0.0,0.0,0.0), vec3(0.4, 0.4, 0.4));
//return mapLadyBug(1.*(p+noise)).x; //錯誤寫法
//return mapLadyBug(1.5*p).x+noise; //正確寫法
}

//=== gradient functions ===
vec3 gradient( in vec3 p ) //尚未normalize
{
    const float d = 0.001;
    vec3 grad = vec3(map(p+vec3(d,0,0))-map(p-vec3(d,0,0)),
                     map(p+vec3(0,d,0))-map(p-vec3(0,d,0)),
                     map(p+vec3(0,0,d))-map(p-vec3(0,0,d)));
    return grad;
}


// === raytrace functions===
float trace(vec3 o, vec3 r, out vec3 p)
{
float d=0.0, t=0.0;
for (int i=0; i<32; ++i)
{
    p= o+r*t;
    d=map(p);
    if(d<0.0) break;
    t += d*0.3; //影響輪廓精準程度
    }
return t;
}

//=== glow functions ===
float glow(float d, float str, float thickness){
    return thickness / pow(d, str);
}
//=== iq’s calc AO ===
float calcAO( in vec3 pos, in vec3 nor )
{
    float ao = 0.0;

    vec3 v = normalize(vec3(0.7,0.5,0.2));
    for( int i=0; i<12; i++ )
    {
        float h = abs(sin(float(i)));
        vec3 kv = v + 2.0*nor*max(0.0,-dot(nor,v));
        ao += clamp( map(pos+nor*0.01+kv*h*0.08)*3.0, 0.0, 1.0 );
        v = v.yzx; //if( (i&2)==2) v.yz *= -1.0;
    }
    ao /= 12.0;
    ao = ao + 2.0*ao*ao;
    return clamp( ao*5.0, 0.0, 1.0 );
}

// === Sky dome ===
vec3 getSkyFBM(vec3 e) {    //二維雲霧
    vec3 f=e;
    float m = 2.0 * sqrt(f.x*f.x + f.y*f.y + f.z*f.z);
    vec2 st= vec2(-f.x/m + .5, -f.y/m + .5);
    //vec3 ret=texture2D(iChannel0, st).xyz;
    float fog= fbm(0.6*st+vec2(-0.2*u_time, -0.02*u_time))*0.5+0.3;
    return vec3(fog);
}

vec3 sky_color(vec3 e) {    //漸層藍天空色
    e.y = max(e.y,0.0);
    vec3 ret;
    ret.x = pow(1.0-e.y,3.0);
    ret.y = pow(1.0-e.y, 1.2);
    ret.z = 0.8+(1.0-e.y)*0.3;    
    return ret;
}

vec3 sky_flame(vec3 e) {    //漸層藍天空色
    e.y = abs(1.-e.y);
    vec3 ret=FlameColour(e.y);   
    return ret;
}

vec2 SphereMap( vec3 ray){      //ray mapping to UV
   vec2 st;
   ray=normalize(ray);
   float radius=length(ray);
   st.y = acos(ray.y/radius) / PI;
   if (ray.z >= 0.0) st.x = acos(ray.x/(radius * sin(PI*(st.y)))) / TWOPI;
   else st.x = 1.0 - acos(ray.x/(radius * sin(PI*(st.y)))) / TWOPI;
   return st;
}
vec4 warpcolor(in vec2 uv, float t){   //Normalized uv[0~1]
            float strength = 0.4;
        vec3 col = vec3(0);
        //pos coordinates (from -1 to 1)
        vec2 pos = uv*2.0-1.0;
            
        //請小心！QC迴圈最好使用int index，float index有可能錯誤！
        for(int i = 1; i < 7; i++){ 
        pos.x += strength * sin(2.0*t+float(i)*1.5 * pos.y)+t*0.5;
        pos.y += strength * cos(2.0*t+float(i)*1.5 * pos.x);}

        //Time varying pixel colour
        col += 0.5 + 0.5*cos(t+pos.xyx+vec3(0,2,4));
        //Gamma
        col = pow(col, vec3(0.4545));
        return vec4(col,1.0) ;
}
vec3 warpSky(vec3 e){   //多彩天空
     vec2 ST=SphereMap(e);
     vec4 color = warpcolor(ST, u_time*0.1);
    return color.xyz;
}

vec3 getSkyALL(vec3 e)
{   
    return getSkyFBM(e);
}

//=== camera functions ===
mat3 setCamera( in vec3 ro, in vec3 ta, float cr )
{
    vec3 cw = normalize(ta-ro);
    vec3 cp = vec3(sin(cr), cos(cr),0.0);
    vec3 cu = normalize( cross(cw,cp) );
    vec3 cv = normalize( cross(cu,cw) );
    return mat3( cu, cv, cw );
}

// math
mat3 fromEuler(vec3 ang) {
    vec2 a1 = vec2(sin(ang.x),cos(ang.x));
    vec2 a2 = vec2(sin(ang.y),cos(ang.y));
    vec2 a3 = vec2(sin(ang.z),cos(ang.z));
    vec3 m0 = vec3(a1.y*a3.y+a1.x*a2.x*a3.x,a1.y*a2.x*a3.x+a3.y*a1.x,-a2.y*a3.x);
    vec3 m1 = vec3(-a2.y*a1.x,a1.y*a2.y,a2.x);
    vec3 m2 = vec3(a3.y*a1.x*a2.x+a1.y*a3.x,a1.x*a3.x-a1.y*a3.y*a2.x,a2.y*a3.y);
    return mat3(m0, m1, m2);
}

// ================
void main()
{
vec2 uv = gl_FragCoord.xy/u_resolution.xy;
uv = uv*2.0-1.0;
uv.x*= u_resolution.x/u_resolution.y;
uv.y*=1.0;//校正 預設值uv v軸朝下，轉成v軸朝上相同於y軸朝上為正
vec2 mouse=(u_mouse.xy/u_resolution.xy)*2.0-1.0;
float angle=mod(u_time*0.5, 6.28);

// camera option1  (模型應在原點，適用於物件)
    vec3 CameraRot=vec3(0.0, -mouse.y, -mouse.x*1.5); 
    vec3 ro= vec3(0.0, 0.0, 2.5)*fromEuler(CameraRot);//CameraPos;
    vec3 ta =vec3(0.0, 0.0, 0.0); //TargetPos; //vec3 ta =float3(CameraDir.x, CameraDir.z, CameraDir.y);//UE座標Z軸在上
    mat3 ca = setCamera( ro, ta, 0.0 );
    vec3 RayDir = ca*normalize(vec3(uv, 2.0));//z值越大，zoom in! 可替換成iMouse.z
    vec3 RayOri = ro;

// camera option2 (攝影機在原點，適用於場景)
/*  
    vec3 CameraRot=vec3(0.0, -iMouse.y, -iMouse.x);
    vec3 RayOri= vec3(0.0, 0.0, 0.0);   //CameraPos;
    vec3 RayDir = normalize(vec3(uv, -1.))*fromEuler(CameraRot);
*/
    
    vec3 p,n;
    float t = trace(RayOri, RayDir, p); //position
    n=normalize(gradient(p)); //normal
    //n+=0.5*normalMap(p,n);    //add normal detail
    float VdotN=dot(-RayDir,n);
    float edge=min(max(smoothstep(-0.388,1.710,VdotN),0.0),1.0);
    //vec3 edge_color=FlameColour(edge);
        
//SHADING
    vec3 result;
    //result=normalize(p);
    //result=(n);
    //result=vec3(t*0.4);
    //result=vec3( 1.0-exp(-t*0.9 ));
    //result=vec3(glow(t,5.0,0.5));
    //result=getSkyFBM(reflect(RayDir,n)); //p or n
    //result=warpSky(reflect(RayDir,n)); //p or n
    //result=vec3(calcAO(p,n));
    //result=n*calcAO(p,n);
    //result=vec3(edge);
    //result=vec3(1.-VdotN);
    result=hatching(p, n, edge);
    
    //result=vec3(glow(edge,1.9,0.05));
    
    
//HDR環境貼圖
    vec3 BG=getSkyALL(RayDir); //或getSkyFBM(RayDir), getSkyALL(RayDir), sky_color(RayDir), warpSky(RayDir)

//亂數作用雲霧(二維)
//float fog= fbm(0.6*uv+vec2(-0.2*u_time, -0.02*u_time))*0.5+0.3;
//vec3 fogFBM=getSkyFBM(reflect(RayDir,n));

gl_FragColor = vec4(vec3(result),1.0);    
//if(t<3.5) gl_FragColor = vec4(vec3(result),1.0); else gl_FragColor = vec4(BG,1.0);
}



















//=== 3d noise functions ===
float hash11(float p) {
    return fract(sin(p * 727.1)*43758.5453123);
}
float hash12(vec2 p) {
    float h = dot(p,vec2(127.1,311.7)); 
    return fract(sin(h)*43758.5453123);
}
vec3 hash31(float p) {
    vec3 h = vec3(1275.231,4461.7,7182.423) * p;    
    return fract(sin(h)*43758.543123);
}
// 3d noise
float noise_3(in vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);  
    vec3 u = f*f*(3.0-2.0*f);    
    vec2 ii = i.xy + i.z * vec2(5.0);
    float a = hash12( ii + vec2(0.0,0.0) );
    float b = hash12( ii + vec2(1.0,0.0) );    
    float c = hash12( ii + vec2(0.0,1.0) );
    float d = hash12( ii + vec2(1.0,1.0) ); 
    float v1 = mix(mix(a,b,u.x), mix(c,d,u.x), u.y);    
    ii += vec2(5.0);
    a = hash12( ii + vec2(0.0,0.0) );
    b = hash12( ii + vec2(1.0,0.0) );    
    c = hash12( ii + vec2(0.0,1.0) );
    d = hash12( ii + vec2(1.0,1.0) );
    float v2 = mix(mix(a,b,u.x), mix(c,d,u.x), u.y);        
    return max(mix(v1,v2,u.z),0.0);
}


//=== 2d noise functions ===

vec2 hash2( vec2 x )            //亂數範圍 [-1,1]
{
    const vec2 k = vec2( 0.3183099, 0.3678794 );
    x = x*k + k.yx;
    return fract( 16.0 * k*fract( x.x*x.y*(x.x+x.y)) );
}
float gnoise( in vec2 p )       //亂數範圍 [-1,1]
{
    vec2 i = floor( p );
    vec2 f = fract( p );
    
    vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( dot( hash2( i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ), 
                            dot( hash2( i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                         mix( dot( hash2( i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ), 
                            dot( hash2( i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
}

float fbm(in vec2 uv)       //亂數範圍 [-1,1]
{
    float f;                //fbm - fractal noise (4 octaves)
    mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );
    f   = 0.5000*gnoise( uv ); uv = m*uv;         
    f += 0.2500*gnoise( uv ); uv = m*uv;
    f += 0.1250*gnoise( uv ); uv = m*uv;
    f += 0.0625*gnoise( uv ); uv = m*uv;
    return f;
}

vec3 smoothSampling2(vec2 uv)
{
    const float T_RES = 32.0;
    return vec3(gnoise(uv*T_RES)); //讀取亂數函式
}

float triplanarSampling(vec3 p, vec3 n)
{
    float fTotal = abs(n.x)+abs(n.y)+abs(n.z);
    return  (abs(n.x)*smoothSampling2(p.yz).x
            +abs(n.y)*smoothSampling2(p.xz).x
            +abs(n.z)*smoothSampling2(p.xy).x)/fTotal;
}

const mat2 m2 = mat2(0.90,0.44,-0.44,0.90);
float triplanarNoise(vec3 p, vec3 n)
{
    const float BUMP_MAP_UV_SCALE = 0.2;
    float fTotal = abs(n.x)+abs(n.y)+abs(n.z);
    float f1 = triplanarSampling(p*BUMP_MAP_UV_SCALE,n);
    p.xy = m2*p.xy;
    p.xz = m2*p.xz;
    p *= 2.1;
    float f2 = triplanarSampling(p*BUMP_MAP_UV_SCALE,n);
    p.yx = m2*p.yx;
    p.yz = m2*p.yz;
    p *= 2.3;
    float f3 = triplanarSampling(p*BUMP_MAP_UV_SCALE,n);
    return f1+0.5*f2+0.25*f3;
}

vec3 normalMap(vec3 p, vec3 n)
{
    float d = 0.005;
    float po = triplanarNoise(p,n);
    float px = triplanarNoise(p+vec3(d,0,0),n);
    float py = triplanarNoise(p+vec3(0,d,0),n);
    float pz = triplanarNoise(p+vec3(0,0,d),n);
    return normalize(vec3((px-po)/d,
                          (py-po)/d,
                          (pz-po)/d));
}


//=== Ladybug distance functions ===
#define MAT_LADY_BODY 3.0
#define MAT_LADY_HEAD 4.0
#define MAT_LADY_LEGS 5.0

//vec2  hash2( vec2 p ) { p=vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))); return fract(sin(p)*18.5453); }
vec3  hash3( float n ) { return fract(sin(vec3(n,n+1.0,n+2.0))*vec3(338.5453123,278.1459123,191.1234)); }
float dot2(in vec2 p ) { return dot(p,p); }
float dot2(in vec3 p ) { return dot(p,p); }

vec2 sdLine( in vec2 p, in vec2 a, in vec2 b )
{
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return vec2( length(pa-h*ba), h );
}
vec2 sdLine( in vec3 p, in vec3 a, in vec3 b )
{
    vec3 pa = p - a;
    vec3 ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return vec2( length(pa-h*ba), h );
}
vec2 sdLineOri( in vec3 p, in vec3 b )
{
    float h = clamp( dot(p,b)/dot(b,b), 0.0, 1.0 );
    
    return vec2( length(p-h*b), h );
}
vec2 sdLineOriY( in vec3 p, in float b )
{
    float h = clamp( p.y/b, 0.0, 1.0 );
    p.y -= b*h;
    return vec2( length(p), h );
}
float sdEllipsoid( in vec3 pos, in vec3 cen, in vec3 rad )
{
    vec3 p = pos - cen;
    float k0 = length(p/rad);
    float k1 = length(p/(rad*rad));
    return k0*(k0-1.0)/k1;
}

float smin( float a, float b, float k )
{
    float h = max(k-abs(a-b),0.0);
    return min(a, b) - h*h*0.25/k;
}

float smax( float a, float b, float k )
{
    float h = max(k-abs(a-b),0.0);
    return max(a, b) + h*h*0.25/k;
}
vec3 rotateX( in vec3 p, float t )
{
    float co = cos(t);
    float si = sin(t);
    p.yz = mat2(co,-si,si,co)*p.yz;
    return p;
}
vec3 rotateY( in vec3 p, float t )
{
    float co = cos(t);
    float si = sin(t);
    p.xz = mat2(co,-si,si,co)*p.xz;
    return p;
}
vec3 rotateZ( in vec3 p, float t )
{
    float co = cos(t);
    float si = sin(t);
    p.xy = mat2(co,-si,si,co)*p.xy;
    return p;
}

vec3 mapLadyBug( vec3 p )
{
    float dBody = sdEllipsoid( p, vec3(0.0), vec3(0.8, 0.75, 1.0) );
    dBody = smax( dBody, -sdEllipsoid( p, vec3(0.0,-0.1,0.0), vec3(0.75, 0.7, 0.95) ), 0.05 );
    dBody = smax( dBody, -sdEllipsoid( p, vec3(0.0,0.0,0.8), vec3(0.35, 0.35, 0.5) ), 0.05 );
    dBody = smax( dBody, sdEllipsoid( p, vec3(0.0,1.7,-0.1), vec3(2.0, 2.0, 2.0) ), 0.05 );
    dBody = smax( dBody, -abs(p.x)+0.005, 0.02 + 0.1*clamp(p.z*p.z*p.z*p.z,0.0,1.0) );

    vec3 res = vec3( dBody, MAT_LADY_BODY, 0.0 );

    // --------
    vec3 hc = vec3(0.0,0.1,0.8);
    vec3 ph = rotateX(p-hc,0.5);
    float dHead = sdEllipsoid( ph, vec3(0.0,0.0,0.0), vec3(0.35, 0.25, 0.3) );
    dHead = smax( dHead, -sdEllipsoid( ph, vec3(0.0,-0.95,0.0), vec3(1.0) ), 0.03 );
    dHead = min( dHead, sdEllipsoid( ph, vec3(0.0,0.1,0.3), vec3(0.15,0.08,0.15) ) );

    if( dHead < res.x ) res = vec3( dHead, MAT_LADY_HEAD, 0.0 );
    
    res.x += 0.0007*sin(150.0*p.x)*sin(150.0*p.z)*sin(150.0*p.y); // iqiq

    // -------------
    
    vec3 k1 = vec3(0.42,-0.05,0.92);
    vec3 k2 = vec3(0.49,-0.2,1.05);
    float dLegs = 10.0;

    float sx = sign(p.x);
    p.x = abs(p.x);
    for( int k=0; k<3; k++ )
    {   
        vec3 q = p;
        q.y -= min(sx,0.0)*0.1;
        if( k==0) q += vec3( 0.0,0.11,0.0);
        if( k==1) q += vec3(-0.3,0.1,0.2);
        if( k==2) q += vec3(-0.3,0.1,0.6);
        
        vec2 se = sdLine( q, vec3(0.3,0.1,0.8), k1 );
        se.x -= 0.015 + 0.15*se.y*se.y*(1.0-se.y);
        dLegs = min(dLegs,se.x);

        se = sdLine( q, k1, k2 );
        se.x -= 0.01 + 0.01*se.y;
        dLegs = min(dLegs,se.x);

        se = sdLine( q, k2, k2 + vec3(0.1,0.0,0.1) );
        se.x -= 0.02 - 0.01*se.y;
        dLegs = min(dLegs,se.x);
    }
    
    if( dLegs<res.x ) res = vec3(dLegs,MAT_LADY_LEGS, 0.0);


    return res;
}

//=== flame color ===
//thanks iq..
// Smooth HSV to RGB conversion 
vec3 hsv2rgb_smooth( in vec3 c )
{
    vec3 rgb = clamp( abs(mod(c.x*6.0+vec3(0.0,4.0,2.0),6.0)-3.0)-1.0, 0.0, 1.0 );

    rgb = rgb*rgb*(3.0-2.0*rgb); // cubic smoothing 

    return c.z * mix( vec3(1.0), rgb, c.y);
}

vec3 hsv2rgb_trigonometric( in vec3 c )
{
    vec3 rgb = 0.5 + 0.5*cos((c.x*6.0+vec3(0.0,4.0,2.0))*3.14159/3.0);

    return c.z * mix( vec3(1.0), rgb, c.y);
}

vec3 FlameColour(float f)
{
    return hsv2rgb_smooth(vec3((f-(2.25/6.))*(1.25/6.),f*1.25+.2,f*.95));
}

//hatching
/*
vec2 hash2( vec2 x )           //亂數範圍 [0,1]
{
    const vec2 k = vec2( 0.3183099, 0.3678794 );
    x = x*k + k.yx;
    return fract( 16.0 * k*fract( x.x*x.y*(x.x+x.y)) );
}
float gnoise( in vec2 p )       //亂數範圍 [0,1]
{
    vec2 i = floor( p );
    vec2 f = fract( p );
    
    vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( dot( hash2( i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ), 
                            dot( hash2( i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                         mix( dot( hash2( i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ), 
                            dot( hash2( i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
}*/

float texh(in vec2 p, in float str)
{
    p*= .7;
    float rz= 1.;
    for (int i=0;i<10;i++)
    {
        //float g = texture(iChannel0,vec2(0.025,.5)*p).x;
        float g = gnoise(vec2(1., 30.2)*p); //亂數範圍 [0,1]
        g = smoothstep(0.-str*0.1,2.3-str*0.1,g);
        rz = min(1.-g,rz);
        p.xy = p.yx;
        p += .07;
        p *= 1.2;
        if (float(i) > str)break;
    }
    return rz*1.05;
}

float cubeproj(in vec3 p, in float str)
{
    float x = texh(p.zy/p.x,str);
    float y = texh(p.xz/p.y,str);
    float z = texh(p.xy/p.z,str);
    p = abs(p);
    if (p.x > p.y && p.x > p.z) return x;
    else if (p.y > p.x && p.y > p.z) return y;
    else return z;
}

float texcube(in vec3 p, in vec3 n, in float str)
{
    float x = texh(p.yz,str);
    float y = texh(p.zx,str);
    float z = texh(p.xy,str);
    n *= n;
    return x*abs(n.x) + y*abs(n.y) + z*abs(n.z);
}

vec3 hatching(vec3 p, vec3 n, float value){        
    vec3 col;
    //float str=10.0;
    float NdotL=(1.0-value)*10.0;
    vec3 col1 = vec3(texcube (p,n,NdotL));
    vec3 col2 = vec3(cubeproj(p,NdotL));
    col = mix(col1,col2, 3.0);
    return col;    
}

