// #version 100

varying float LightIntensity; 
varying vec4  Color;

uniform float UserParameter;

void main()
{
    float intensity = min(1., max(0., LightIntensity)+0.2);
    gl_FragColor = vec4(Color.rgb * intensity, 1.);
}
