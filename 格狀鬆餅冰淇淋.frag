// Author:
// Title:

#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;

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
    const float e = -6.0;
    return log(exp(d1*e)+exp(d2*e))/e;
}


float drop(vec3 p ){
    float h1 = -sin(0.25);
    float h2 = cos(0.25+.1);
    float drop = length(p+vec3(5.,1.0,0.0)*h1)-.4;
    drop = smin(drop,length(p+vec3(.1,.8,0)*h2)-1.168);
    return drop;
    
}

float ripple(vec3 p){
   float l = pow(dot(p.xz,p.xz),0.8);
    float ripple = p.y+0.8+.3*sin(l*3.-u_time+0.5)/(1.0+l); 
    return ripple;
}
//Ripple and drop distance function
float dist(vec3 p)
{
   float ripple=ripple(p);
   float drop=drop(p);
    return smin(ripple,drop);
}
float sdFloor(vec3 p) {
  return p.y + 1.0;
}
float map(in vec3 p)
{
float Rightdist=dist(p-vec3(-2.5,0.0,-2.0));
float Leftdist=dist(p);
//float Leftdist=dist(p-vec3(-2.5,0.0,-2.0));
//float bump=0.020 * (noise_3(p*6.0)*2.0-1.0);
//vec3 p1 = p + bump;
//return sdSphere(p1+vec3(0.,0.,0.0), 0.5);
//return dist(p);
//return sdBox(p+vec3(0.0,0.0,0.0), vec3(0.4, 0.4, 0.4));
//return udRoundBox(p+vec3(0.0,0.0,0.0), vec3(0.3, 0.3, 0.3), 0.1);
//return mix(Rightdist,Leftdist,3.0);
return min(Rightdist,Leftdist);
 //float res = min(Rightdist,Leftdist);
  //res = min(res, sdFloor(p));
 // return res;   
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
        float s = dist(m.xyz);
        m += vec4(d,1)*s;
        
        if (s<.01 || m.w>20.) break;
    }
    return m;
}

void main( )
{
    vec2 res = u_resolution.xy;
    vec3 col = vec3(0);
    vec3 pos = vec3(.05*cos(u_time),.1*sin(u_time),-6.0);
    vec3 lig = sqrt(vec3(.3,.5,.2));
   
    //Sample
     //int num_ripples = 10.0;
    for(float x = 0.;x<AA;x++)
    for(float y = 0.;y<AA;y++)
    {
        vec3 ray = normalize(vec3(gl_FragCoord.xy-res/2.+vec2(x,y)/AA,res.y));
        vec4 mar = march(pos,ray);
        vec3 nor = normal(mar.xyz);
        vec3 ref = refract(ray,nor,.75);
        float r = smoothstep(.8,1.,dot(reflect(ray,nor),lig));
        float l = 1.-dot(ray,nor);
        vec3 wat = background(ref)+.3*r*l*l;
        vec3 bac = background(ray)*.5+.5;

        float fade = pow(min(mar.w/20.,1.),.3);
        col += mix(wat,bac,fade);
    }
    col /= AA*AA;

    gl_FragColor = vec4(col*col,1);
}

