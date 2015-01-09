// #version 100

varying vec4  Color;

uniform float Time;
uniform float UserParameter;


void main()
{
    Color = gl_Color;

    gl_Position     = gl_ModelViewProjectionMatrix * vec4(vec3(gl_Vertex.xyz/gl_Vertex.w), 1.);
}
