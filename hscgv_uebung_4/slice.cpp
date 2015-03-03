/** \file
 * \brief Coloured 2D-slice
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 4 - "LBM"
 *
 * Created Martin Aumueller <aumueller@uni-koeln.de>
 */

#include "slice.h"
#include "lbm.h"

#include <iostream>
#include <cassert>

#include <osg/Array>
#include <osg/Texture2D>
#include <osg/Texture1D>

static const int DataTex = 0;
static const int TfTex = 1;

static const char *classfrag =
"uniform sampler2D dataTex;\n"
"uniform sampler1D tfTex;\n"
"\n"
"void main(void)\n"
"{\n"
"    float x = texture2D(dataTex, gl_TexCoord[0].st).x;\n"
"    vec4 c = texture1D(tfTex, x);\n"
"gl_FragColor = c * gl_Color;\n"
"}\n";

// inspired by http://www.cs.unm.edu/~kmorel/documents/ColorMaps/ColorMapsExpanded.pdf
static unsigned char tfdata[8][4] =
{
    { 59, 76, 192, 255 },
    { 98, 130, 234, 255 },
    { 141, 176, 254, 255 },
    { 184, 208, 249, 255 },
    // { 221, 221, 221 },
    { 245, 196, 173, 255 },
    { 244, 154, 123, 255 },
    { 222, 96, 77, 255 },
    { 180, 4, 38, 255 },
};

Slice::Slice(LBMD3Q19 *lbm, int axis)
: m_lbm(lbm)
, m_axis(axis)
{
    addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 4));

    // geometry and normals
    osg::Vec3Array *vert = new osg::Vec3Array;
    osg::Vec3Array *norm = new osg::Vec3Array;
    switch(axis)
    {
        case 0:
            m_width = m_lbm->getDimension(1);
            m_height = m_lbm->getDimension(2);
            vert->push_back(osg::Vec3(0., 0., 0.));
            vert->push_back(osg::Vec3(0., 0., m_height));
            vert->push_back(osg::Vec3(0., m_width, m_height));
            vert->push_back(osg::Vec3(0., m_width, 0.));
            norm->push_back(osg::Vec3(-1., 0., 0.));
            break;
        case 1:
            m_width = m_lbm->getDimension(0);
            m_height = m_lbm->getDimension(2);
            vert->push_back(osg::Vec3(0., 0., 0.));
            vert->push_back(osg::Vec3(0., 0., m_height));
            vert->push_back(osg::Vec3(m_width, 0., m_height));
            vert->push_back(osg::Vec3(m_width, 0., 0.));
            norm->push_back(osg::Vec3(0., 1., 0.));
            break;
        case 2:
            m_width = m_lbm->getDimension(0);
            m_height = m_lbm->getDimension(1);
            vert->push_back(osg::Vec3(0., 0., 0.));
            vert->push_back(osg::Vec3(0., m_height, 0.));
            vert->push_back(osg::Vec3(m_width, m_height, 0.));
            vert->push_back(osg::Vec3(m_width, 0., 0.));
            norm->push_back(osg::Vec3(0., 0., -1.));
            break;
        default:
            assert(axis >= 0 && axis <= 2);
    }
    setVertexArray(vert);
    setNormalArray(norm);
    setNormalBinding(osg::Geometry::BIND_OVERALL);

    m_width2 = osg::Image::computeNearestPowerOfTwo(m_width, 1.);
    m_height2 = osg::Image::computeNearestPowerOfTwo(m_height, 1.);

    // tex coords
    osg::Vec2Array *tc = new osg::Vec2Array;
    tc->push_back(osg::Vec2(0., 0.));
    tc->push_back(osg::Vec2(0., 1.*m_height/m_height2));
    tc->push_back(osg::Vec2(1.*m_width/m_width2, 1.*m_height/m_height2));
    tc->push_back(osg::Vec2(1.*m_width/m_width2, 0.));
    setTexCoordArray(DataTex, tc);

    // color (used for fixed-function lighting)
    osg::Vec4Array *col = new osg::Vec4Array;
    col->push_back(osg::Vec4(1., 1., 1., 1.));
    setColorArray(col);
    setColorBinding(osg::Geometry::BIND_OVERALL);

    // texture for data
    osg::StateSet *state = getOrCreateStateSet();
    m_dataTex =  new osg::Texture2D;
    m_dataTex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
    m_dataTex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::CLAMP_TO_EDGE);
    m_dataTex->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture::LINEAR);
    m_dataTex->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture::LINEAR);
    state->setTextureAttributeAndModes(DataTex, m_dataTex, osg::StateAttribute::ON);
    m_img = new osg::Image;
    m_img->setDataVariance(osg::Object::DYNAMIC);
    m_dataTex->setImage(m_img);

    // texture for transfer function
    m_tfTex = new osg::Texture1D;
    m_tfTex->setWrap(osg::Texture1D::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
    m_tfTex->setFilter(osg::Texture1D::MIN_FILTER, osg::Texture::LINEAR);
    m_tfTex->setFilter(osg::Texture1D::MAG_FILTER, osg::Texture::LINEAR);

    osg::Image *tfimg = new osg::Image;
    tfimg->setImage(8, 1, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, &tfdata[0][0], osg::Image::NO_DELETE);
    m_tfTex->setImage(tfimg);
    state->setTextureAttributeAndModes(TfTex, m_tfTex, osg::StateAttribute::ON|osg::StateAttribute::PROTECTED);

    // classification shader
    osg::Program *prog = new osg::Program;
    prog->addShader(new osg::Shader(osg::Shader::FRAGMENT, classfrag));
    osg::Uniform *dataTex = new osg::Uniform(osg::Uniform::SAMPLER_2D, "dataTex", 1);
    dataTex->set(DataTex);
    state->addUniform(dataTex);
    osg::Uniform *tfTex = new osg::Uniform(osg::Uniform::SAMPLER_1D, "tfTex", 1);
    tfTex->set(TfTex);
    state->addUniform(tfTex);
    state->setAttributeAndModes(prog, osg::StateAttribute::ON);

    setStateSet(state);
}

Slice::~Slice()
{
}

void Slice::update(int slice, DataValue value)
{
    unsigned char *data = new unsigned char[m_height2*m_width2];

    double minD = m_lbm->minDensity();
    double maxD = m_lbm->maxDensity();
    double maxV = m_lbm->maxVelocity();
    for(int j=0; j<m_height; ++j)
    {
        for(int i=0; i<m_width; ++i)
        {
            double c = 0.;
            double d = 0.;
            LBMD3Q19::Vector v;
            switch(m_axis)
            {
                case 0:
                    d = m_lbm->density(slice, i, j);
                    v = m_lbm->velocity(slice, i, j);
                    break;
                case 1:
                    d = m_lbm->density(i, slice, j);
                    v = m_lbm->velocity(i, slice, j);
                    break;
                case 2:
                    d = m_lbm->density(i, j, slice);
                    v = m_lbm->velocity(i, j, slice);
                    break;
            }

            switch(value)
            {
                case Density:
                    c = (d-minD)/(maxD-minD);
                    break;
                case Velocity:
                    c = sqrtf(v.e[0]*v.e[0] + v.e[1]*v.e[1] + v.e[2]*v.e[2]);
                    c /= maxV;
                    break;
            }

            if(c < 0.)
                c=0.;
            if(c > 1.)
                c=1.;
            data[i+j*m_width2] = c * 255.;
        }
    }

    if(m_height < m_height2)
    {
        for(int i=0; i<m_width2; ++i)
        {
            data[i+m_height*m_width2] = data[i+(m_height-1)*m_width2];
        }
    }
    if(m_width < m_width2)
    {
        for(int j=0; j<m_height2; ++j)
        {
            data[m_width+j*m_width2] = data[m_width-1+j*m_width2];
        }
    }

    m_img->setImage(m_width2, m_height2, 1, GL_LUMINANCE, GL_LUMINANCE, GL_UNSIGNED_BYTE, data, osg::Image::USE_NEW_DELETE);
}
