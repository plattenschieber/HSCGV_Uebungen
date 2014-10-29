// #version 100

varying float LightIntensity;
varying vec4  Color;

uniform float Time;
uniform float UserParameter;

const float M_PI = 3.14159265;


void main()
{
    Color = gl_Color;

    float Scale = UserParameter*0.01;

    vec3 vertex = vec3(gl_Vertex.xyz/gl_Vertex.w);
    vertex.y += Scale*sin(M_PI*vertex.x*10.+Time*1.1)*sin(M_PI*vertex.z*10.+Time*1.1);
    vertex.x += Scale*sin(M_PI*vertex.y*10.+Time)*sin(M_PI*vertex.z*10.+Time);
    vertex.z += Scale*sin(M_PI*vertex.x*10.+Time*1.3)*sin(M_PI*vertex.x*10.+Time*1.3);

    gl_Position     = gl_ModelViewProjectionMatrix * vec4(vertex, 1.);

    vec3 EyePosition = vec3(gl_ModelViewMatrix * gl_Vertex);
    vec3 norm      = normalize(vec3(gl_NormalMatrix * gl_Normal));
    LightIntensity  = clamp(dot(normalize(vec3(gl_LightSource[0].position) - EyePosition), norm), 0., 1.);
}
