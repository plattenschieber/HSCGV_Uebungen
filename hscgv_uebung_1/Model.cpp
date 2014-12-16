/** \file
 * \brief This module contains the model loading and drawing routines.
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 1 - "Ui, OpenGL!"
 *
 * Created by Martin Aumueller <aumueller@uni-koeln.de>
 */

#include <cstdlib>
#include <cmath>
#include <cfloat>

#include <QDebug>
#include <QFile>

#include "Model.h"
#include "ply.h"

#ifdef _WIN32
static float fmin(float a, float b)
{
    return a<b ? a : b;
}

static float fmax(float a, float b)
{
    return a>b ? a : b;
}
#endif

#ifdef NDEBUG
#define CHECKGL
#else
static bool checkGL(const char *label, int line)
{
    GLenum err = glGetError();
    if(err == GL_NO_ERROR)
        return true;

    qDebug() << "GL error -" << label << line << (char*)gluErrorString(err);
    return false;
}
#ifdef _WIN32
#define CHECKGL do { checkGL(__FUNCTION__, __LINE__); } while(0)
#else
#define CHECKGL do { checkGL(__PRETTY_FUNCTION__, __LINE__); } while(0)
#endif
#endif


Model::Model(QObject *parent)
: QObject(parent)
, m_ply(NULL)
{
    memset(m_bo, '\0', sizeof(m_bo));
    memset(m_center, '\0', sizeof(m_center));
    m_size = 0.;
}

Model::~Model()
{
    release();

    if(m_ply)
        free_ply(m_ply);
}

void Model::getCenter(GLfloat *x, GLfloat *y, GLfloat *z) const
{
    *x = m_center[0];
    *y = m_center[1];
    *z = m_center[2];
}

GLfloat Model::size() const
{
    return m_size;
}

// ------------------------- Drawing Methods ---------------------------------

// draw model using old-fashioned glBegin()/glEnd()
void Model::drawImmediate(bool smooth)
{
    for(std::vector<Face>::iterator it = m_face.begin();
            it != m_face.end();
            ++it)
    {
        const Face &f = *it;
        if(smooth)
        {
            glBegin(GL_POLYGON);
            for(int i=0; i<f.nverts; ++i)
            {
                const Vertex &v = m_vert[f.verts[i]];
                float linv = 1.f/sqrtf(v.nx*v.nx + v.ny*v.ny + v.nz*v.nz);
                glNormal3f(v.nx*linv, v.ny*linv, v.nz*linv);
                glVertex3f(v.x, v.y, v.z);
            }
            glEnd();
        }
        else
        {
            glNormal3f(f.nx, f.ny, f.nz);
            glBegin(GL_POLYGON);
            for(int i=0; i<f.nverts; ++i)
            {
                const Vertex &v = m_vert[f.verts[i]];
                glVertex3f(v.x, v.y, v.z);
            }
            glEnd();
        }
    }
}

// draw model using vertex arrays
void Model::drawArray(bool smooth)
{
    //! enble and specify pointers to vertex arrays
    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);
    //! the last argument needs special treatment
    glNormalPointer(GL_FLOAT, 0, &m_vertexNormalArray[0]);
    glVertexPointer(3, GL_FLOAT, 0, &m_vertexArray[0]);
    glMultiDrawElements(GL_POLYGON, &m_primitiveSizeArray[0], GL_UNSIGNED_INT,
                        &m_vertexIndexStartArray[0], m_face.size());
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
}

// draw model using vertex buffer objects (VBOs)
void Model::drawVBO(bool smooth)
{
    /* TODO
      * bind buffers with appropriate targets
      * use gl...Pointer to make data available
      * enable vertex and normal arrays
      * call glMultiDrawElements
      * unbind all buffers
      * disable vertex arrays
      */

    //!vertices
    glBindBuffer(GL_ARRAY_BUFFER, m_bo[BO_VERTEX]);
    glVertexPointer(3, GL_FLOAT, 0, (void*)0);

    //!Normals
    glBindBuffer(GL_ARRAY_BUFFER, m_bo[BO_VERTEX_NORMAL]);
    glNormalPointer(GL_FLOAT, 0, (void*)0 );

    //! enable vertex and normal array like in drawArray
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);

    //!Indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_bo[BO_VERTEX_INDEX]);

    //! the last argument needs special treatment
#ifdef DEBUG
    for (uint i=0; i<m_primitiveSizeArray.size(); i++){
        if (m_primitiveSizeArray[i] > 0)
            glDrawElements(GL_POLYGON, m_primitiveSizeArray[i], GL_UNSIGNED_INT, m_primitiveOffsetArray[i]);
    }
#else
    glMultiDrawElements(GL_POLYGON, &m_primitiveSizeArray[0], GL_UNSIGNED_INT,
                        &m_primitiveOffsetArray[0], m_face.size());
    CHECKGL;
#endif

    //!Clean up
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

// generate data used for vertex array and vertex buffer object GL modes
void Model::computeVertexArrayData()
{
    release();

    int iVerts = 0;
    //! iterate over all faces
    for(std::vector<Face>::iterator it = m_face.begin();
            it != m_face.end();
            ++it)
    {
        const Face &f = *it;
        //! store number of vertices for each face
        m_primitiveSizeArray.push_back(f.nverts);

        //! for each vertex in each face
        for(int i=0; i<f.nverts; ++i)
        {
            //! store the coordinates of this vertex
            m_vertexArray.push_back(m_vert[f.verts[i]].x);
            m_vertexArray.push_back(m_vert[f.verts[i]].y);
            m_vertexArray.push_back(m_vert[f.verts[i]].z);

            //! store the normalised? vertex normals (which are the same as the normals
            //! of the face they are belonging to) [we don't use smoothness here]
            m_vertexNormalArray.push_back(f.nx);
            m_vertexNormalArray.push_back(f.ny);
            m_vertexNormalArray.push_back(f.nz);

            //!  store its index
            m_vertexIndexArray.push_back(iVerts++);
        }
        //! store the face normals
        m_faceNormalArray.push_back(f.nx);
        m_faceNormalArray.push_back(f.ny);
        m_faceNormalArray.push_back(f.nz);
    }

    int currentIndex = 0;
    //! iterate again over all faces
    for(std::vector<Face>::iterator it = m_face.begin();
            it != m_face.end();
            ++it)
    {
        const Face &f = *it;
        //! store a pointer to the first vertex in the array of vertex indices
        m_vertexIndexStartArray.push_back(&m_vertexIndexArray[currentIndex]);
        //! store the offset of the first vertex into the array of vertex indices
        m_primitiveOffsetArray.push_back((GLuint *)0 +currentIndex);
        currentIndex += f.nverts;
    }
}

// buffer data for VBO GL mode
void Model::bufferArrayData()
{
    /* TODO
      * create buffer objects
      * bind each buffer to the appropriate target
      * buffer appropriate data
      * bind 0 to the target
      */

    //! generate new buffer objects and store the generated IDs
    glGenBuffers(NUM_BOS, m_bo);

    //! bind the buffer objects for vertex, normal and indices data and copy data to the BOs

    //! vertices
    glBindBuffer(GL_ARRAY_BUFFER, m_bo[BO_VERTEX]);
    glBufferData(GL_ARRAY_BUFFER, m_vertexArray.size()*sizeof(GLfloat), &m_vertexArray[0], GL_STATIC_DRAW);

    //! vertex and face Normals
    glBindBuffer(GL_ARRAY_BUFFER, m_bo[BO_VERTEX_NORMAL]);
    glBufferData(GL_ARRAY_BUFFER, m_vertexNormalArray.size()*sizeof(GLfloat), &m_vertexNormalArray[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, m_bo[BO_FACE_NORMAL]);
    glBufferData(GL_ARRAY_BUFFER, m_faceNormalArray.size()*sizeof(GLfloat), &m_faceNormalArray[0], GL_STATIC_DRAW);

    //! offsets into vertices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_bo[BO_VERTEX_INDEX]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_primitiveOffsetArray.size()*sizeof(GLuint), &m_primitiveOffsetArray[0], GL_STATIC_DRAW);

    //! unbind the buffer objects
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void Model::release()
{
    glDeleteBuffers(NUM_BOS, m_bo);
    memset(m_bo, '\0', sizeof(m_bo));

    m_vertexArray.clear();
    m_vertexNormalArray.clear();
    m_vertexIndexArray.clear();
    m_faceNormalArray.clear();
    m_primitiveSizeArray.clear();
    m_vertexIndexStartArray.clear();
}

// read model using standard ply reader
void Model::read(const QString &model)
{
    static PlyProperty vert_props[] = { /* list of property information for a vertex */
        {"x", Float32, Float32, offsetof(Vertex,x), 0, 0, 0, 0},
        {"y", Float32, Float32, offsetof(Vertex,y), 0, 0, 0, 0},
        {"z", Float32, Float32, offsetof(Vertex,z), 0, 0, 0, 0},
        {"nx", Float32, Float32, offsetof(Vertex,nx), 0, 0, 0, 0},
        {"ny", Float32, Float32, offsetof(Vertex,ny), 0, 0, 0, 0},
        {"nz", Float32, Float32, offsetof(Vertex,nz), 0, 0, 0, 0},
    };

    static PlyProperty face_props[] = { /* list of property information for a face */
        {"vertex_indices", Int32, Int32, offsetof(Face,verts),
            1, Uint8, Uint8, offsetof(Face,nverts)},
    };

    if(m_ply)
    {
        free_ply(m_ply);
        m_ply = NULL;
    }

    m_vert.clear();
    m_face.clear();


    int has_nx = 0, has_ny = 0, has_nz = 0;

    QFile file(model);
    if(!file.open(QIODevice::ReadOnly))
	{
        QString msg = "Could not open " + model;
        qDebug() << msg;
        emit info(msg);
        return;
	}
    FILE *fp = fdopen(file.handle(), "r");
    if(!fp)
    {
        QString msg = "Could not open " + model;
        qDebug() << msg;
        emit info(msg);
        return;
    }

    float min[3] = { FLT_MAX,  FLT_MAX,  FLT_MAX};
    float max[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

    // transfer ownership of fp to PLY reader
    m_ply = read_ply(fp);
    if(!m_ply)
    {
        QString msg = "Could not interpret " + model;
        qDebug() << msg;
        emit info(msg);
        return;
    }

    /* examine each element type that is in the file (vertex, face) */

    for (int i = 0; i < m_ply->num_elem_types; i++) {
        int elem_count = 0;

        /* prepare to read the i'th list of elements */
        const char *elem_name = setup_element_read_ply (m_ply, i, &elem_count);

        if (equal_strings ("vertex", elem_name)) {
            /* set up for getting vertex elements */
            /* (we want x,y,z) */

            setup_property_ply (m_ply, &vert_props[0]);
            setup_property_ply (m_ply, &vert_props[1]);
            setup_property_ply (m_ply, &vert_props[2]);

            /* we also want normal information if it is there (nx,ny,nz) */

            for (int j = 0; j < m_ply->elems[i]->nprops; j++) {
                PlyProperty *prop = m_ply->elems[i]->props[j];
                if (equal_strings ("nx", prop->name)) {
                    setup_property_ply (m_ply, &vert_props[3]);
                    has_nx = 1;
                }
                if (equal_strings ("ny", prop->name)) {
                    setup_property_ply (m_ply, &vert_props[4]);
                    has_ny = 1;
                }
                if (equal_strings ("nz", prop->name)) {
                    setup_property_ply (m_ply, &vert_props[5]);
                    has_nz = 1;
                }
            }

            /* also grab anything else that we don't need to know about */
            (void)get_other_properties_ply (m_ply,
                    offsetof(Vertex,other_props));

            /* grab the vertex elements and store them in our list */

            for (int j = 0; j < elem_count; j++) {
                Vertex v;
                get_element_ply (m_ply, &v);

                min[0] = fmin(v.x, min[0]);
                min[1] = fmin(v.y, min[1]);
                min[2] = fmin(v.z, min[2]);
                max[0] = fmax(v.x, max[0]);
                max[1] = fmax(v.y, max[1]);
                max[2] = fmax(v.z, max[2]);


                m_vert.push_back(v);
            }
        }
        else if (equal_strings ("face", elem_name)) {
            /* set up for getting face elements */
            /* (all we need are vertex indices) */

            setup_property_ply (m_ply, &face_props[0]);
            (void)get_other_properties_ply (m_ply,
                    offsetof(Face,other_props));

            /* grab all the face elements and place them in our list */

            for (int j = 0; j < elem_count; j++) {
                Face f;
                get_element_ply (m_ply, &f);
                if(f.nverts >= 3)
                {
                    // compute face normals
                    Vertex v[3];
                    for(int i=0; i<3; ++i)
                        v[i] = m_vert[f.verts[i]];
                    f.nx = (v[1].y-v[0].y)*(v[2].z-v[0].z)-(v[1].z-v[0].z)*(v[2].y-v[0].y);
                    f.ny = (v[1].z-v[0].z)*(v[2].x-v[0].x)-(v[1].x-v[0].x)*(v[2].z-v[0].z);
                    f.nz = (v[1].x-v[0].x)*(v[2].y-v[0].y)-(v[1].y-v[0].y)*(v[2].x-v[0].x);
                    const float linv = 1.f/sqrtf(f.nx*f.nx + f.ny*f.ny + f.nz*f.nz);
                    f.nx *= linv;
                    f.ny *= linv;
                    f.nz *= linv;
                }
                else
                {
                    f.nx = f.ny = f.nz = 0.f;
                }
                m_face.push_back(f);
            }
        }
        else  /* all non-vertex and non-face elements are grabbed here */
            get_other_element_ply (m_ply);
    }

    /* close the file */
    /* (we won't free up the memory for m_ply because we will use it */
    /*  to help describe the file that we will write out) */

    // don't close it yet: done by QFile dtor
	//close_ply (m_ply);

    float d2 = 0.f;
    for(int i=0; i<3; ++i)
    {
        float d = max[i] - min[i];
        m_center[i] = 0.5*(max[i] + min[i]);
        d2 += d*d;
    }
    m_size = sqrtf(d2);

    if(!(has_nx && has_ny && has_nz))
        computeVertexNormals();

    computeVertexArrayData();
    bufferArrayData();

    QString msg;
    msg.sprintf("%s (%d vertices, %d faces%s)", model.toLocal8Bit().constData(), (int)m_vert.size(), (int)m_face.size(),
                has_nx&&has_ny&&has_nz ? "" : ", no normals");
    emit info(msg);
}

void Model::computeVertexNormals()
{
    std::vector<std::vector<int> > vertexfaces; // store, in which faces a vertex is used
    for(size_t i=0; i<m_face.size(); ++i)
    {
        const Face &f = m_face[i];
        for(size_t j=0; j<f.nverts; ++j)
        {
            const size_t v = f.verts[j];
            if(vertexfaces.size() < v+1)
                vertexfaces.resize(v+1);
            vertexfaces[v].push_back(i);
        }
    }

    // now, faces[i] contains an std::vector of indices of all the faces m_vert[i] belongs to


    // loop over all vertices and all faces it belongs to
    for(size_t i=0; i<vertexfaces.size(); ++i)
    {
        Vertex &v = m_vert[i]; 
        float n[3] = {0.f, 0.f, 0.f};
        /* TODO
          * loop over all faces v belongs to
          ** add each facenormal to n
          * normalise computed vertex normal n
         */

        // store result of averaging
        v.nx = n[0];
        v.ny = n[1];
        v.nz = n[2];
    }
}
