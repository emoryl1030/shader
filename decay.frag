// Author:
// Title:

#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;
float noise_3(in vec3 p); 
//Anti-Aliasing (SSAA). Use 1.0 on slower computers
#define AA 2.

//Background gradient
vec3 background(vec3 d)
{
    float light = dot(d,sqrt(vec3(.3,.5,.2)));
    
    return vec3(max(light*.5+.5,.0));
}
//Smooth minimum (based off IQ's work)
float smin(float d1, float d2)
{
    const float e = -6.;
    return log(exp(d1*e)+exp(d2*e))/e;
}
//Ripple and drop distance function
float dist(vec3 p)
{
   
    float l = pow(dot(p.xz,p.xz),0.784);
    float attenuation = exp(-l * 0.023);
    float ripple = p.y*attenuation+0.8+0.4*sin(l*3.888-0.25+.5)/(1.0+l) ;
    float h1 = -sin(0.25);
    float h2 = cos(0.25);
    float drop = length(p+vec3(3.0,1.0,0.0)*h1)-0.984;
    drop = smin(drop,length(p+vec3(.1,.8,0)*h2)-1.280);
    return smin(ripple,drop);
}
float map(in vec3 p)
{
float bump=0.002 * (noise_3(p*60.0)*2.0-1.0);//前面0.002影響線條，後面數值影響玻璃程度
vec3 p1 = p + bump;
//return sdSphere(p1+vec3(0.,0.,0.0), 0.5);
//return sdTorus(p+vec3(0.,0.,0.0),vec2(0.4,0.2))+bump;
return dist(p);
//return udRoundBox(p+vec3(0.0,0.0,0.0), vec3(0.3, 0.3, 0.3), 0.1);
}

//Typical SDF normal function
vec3 normal(vec3 p)
{
    vec2 e = vec2(1,-1)*.01;
    
    return normalize(map(p-e.yxx)*e.yxx+map(p-e.xyx)*e.xyx+
    map(p-e.xxy)*e.xxy+map(p-e.y)*e.y);
}
//Basic raymarcher
vec4 march(vec3 p, vec3 d)
{
    vec4 m = vec4(p,0);
    for(int i = 0; i<99; i++)
    {
        float s = map(m.xyz);
        m += vec4(d,1)*s;
        
        if (s<.01 || m.w>20.) break;
    }
    return m;
}

// === raytrace functions===
float traceInner(vec3 o, vec3 r, out vec3 p)
{
float d=0.0, t=0.01;//t偏移一點影響Ｐ，不然一直在原點踩不出去
for (int i=0; i<16; ++i)//節省效能
{
	p= o+r*t;
	d=-map(p);//or動手腳負值
	if(d<0.001||t>10.0) break;//終止條件可以設得很精準
	t += d*0.5; //or動手腳負值
	}
return t;
}

void main( )
{
    vec2 res = u_resolution.xy;
    vec3 col = vec3(0);
    
    vec3 pos = vec3(.05*cos(u_time),.1*sin(u_time),-4);
    vec3 lig = sqrt(vec3(.3,.5,.2));
    
    //Sample
    for(float x = 0.;x<AA;x++)
    for(float y = 0.;y<AA;y++)
    {
        vec3 ray = normalize(vec3(gl_FragCoord.xy-res/2.0+vec2(x,y)/AA,res.y));
        vec4 mar = march(pos,ray);
        vec3 nor = normal(mar.xyz);
        vec3 ref = refract(ray,nor,0.910);
        float r = smoothstep(.8,1.,dot(reflect(ray,nor),lig));
        float l = 1.-dot(ray,nor);
        vec3 wat = background(ref)+.3*r*l*l;
        vec3 bac = background(ray)*0.5+.5;

        float fade = pow(min(mar.w/20.,1.),.3);
        col += mix(wat,bac,fade);
    }
    col /= AA*AA;

    gl_FragColor = vec4(col*col,1);
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
