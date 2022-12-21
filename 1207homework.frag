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
    const float e = -6.;
    return log(exp(d1*e)+exp(d2*e))/e;
}
//Ripple and drop distance function
float dist(vec3 p)
{
    float l = pow(dot(p.xz,p.xz),0.8);
    float ripple = p.y+0.8+.4*sin(l*3.-u_time+.5)/(1.+l);
    
    float h1 = -sin(u_time);
    float h2 = cos(u_time+.1);
    float drop = length(p+vec3(0,1.2,0)*h1)-.4;
    drop = smin(drop,length(p+vec3(.1,.8,0)*h2)-.2);
    return smin(ripple,drop);
}
//Typical SDF normal function
vec3 normal(vec3 p)
{
    vec2 e = vec2(1,-1)*.01;
    
    return normalize(dist(p-e.yxx)*e.yxx+dist(p-e.xyx)*e.xyx+
    dist(p-e.xxy)*e.xxy+dist(p-e.y)*e.y);
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

// === raytrace functions===
float traceInner(vec3 o, vec3 r, out vec3 p)
{
float d=0.0, t=0.01;//t偏移一點影響Ｐ，不然一直在原點踩不出去
for (int i=0; i<16; ++i)//節省效能
{
	p= o+r*t;
	d=-dist(p);//or動手腳負值
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


