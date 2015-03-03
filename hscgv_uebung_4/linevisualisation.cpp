/** \file
 * \brief Line visualisation
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 4 - "LBM"
 *
 * Created Martin Aumueller <aumueller@uni-koeln.de>
 */

#include "linevisualisation.h"
#include "lbm.h"

#include <osg/Geode>
#include <osg/BlendFunc>

LineVisualisation::LineVisualisation(LBMD3Q19 *lbm)
: m_lbm(lbm)
, m_width(lbm->getDimension(0))
, m_height(lbm->getDimension(1))
, m_depth(lbm->getDimension(2))
{
    m_geom = new osg::Geometry;
    m_geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, 2*m_width*m_height*m_depth));
    addDrawable(m_geom);

    osg::Vec4Array *col = new osg::Vec4Array;
    m_vert = new osg::Vec3Array;
    m_vert->setDataVariance(osg::Object::DYNAMIC);
    for(int k=0; k<m_depth; ++k)
    {
        for(int j=0; j<m_height; ++j)
        {
            for(int i=0; i<m_width; ++i)
            {
                m_vert->push_back(osg::Vec3(i+0.5, j+0.5, k+0.5));
                m_vert->push_back(osg::Vec3(i+0.5, j+0.5, k+0.5));
                col->push_back(osg::Vec4(0., 1., 0., 0.));
                col->push_back(osg::Vec4(0., 1., 0., 1.));
            }
        }
    }
    m_geom->setVertexArray(m_vert);

    m_geom->setColorArray(col);
    m_geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    osg::StateSet *state = getOrCreateStateSet();
    state->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    state->setMode(GL_BLEND, osg::StateAttribute::ON);
    state->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    osg::BlendFunc* blendFunc = new osg::BlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    state->setAttribute(blendFunc);
    setStateSet(state);
}

LineVisualisation::~LineVisualisation()
{
}

void LineVisualisation::update()
{
    size_t idx = 0;
    for(int k=0; k<m_depth; ++k)
    {
        for(int j=0; j<m_height; ++j)
        {
            for(int i=0; i<m_width; ++i)
            {
                LBMD3Q19::Vector v = m_lbm->velocity(i, j, k);
                (*m_vert)[idx+1] = (*m_vert)[idx] + osg::Vec3(v[0], v[1], v[2]) * 5.;
                idx += 2;
            }
        }
    }
    m_geom->setVertexArray(m_vert);
    m_geom->dirtyDisplayList();
}
