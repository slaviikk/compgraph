#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <map>
#include <random>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
// КРИТЕРИЙ: ОСВЕЩЕНИЕ
// Структуры для освещения
struct DirLight {
    glm::vec3 direction;
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
};

struct PointLight {
    glm::vec3 position;
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
    float constant;
    float linear;
    float quadratic;
};
// КРИТЕРИЙ: АНИМАЦИЯ СЦЕНЫ
// Структура для частиц дыма
struct SmokeParticle {
    glm::vec3 position;
    glm::vec3 velocity;
    float life;
    float scale;
};
// КРИТЕРИЙ: БАЗОВАЯ СЦЕНА
// Структуры для рендеринга объектов
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoords;
};

struct Texture {
    GLuint id;
    std::string type;
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture> textures;
    GLuint VAO, VBO, EBO;

    void setupMesh() {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex),
                    &vertices[0], GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
                    &indices[0], GL_STATIC_DRAW);

        // Атрибуты вершин
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                             (void*)offsetof(Vertex, normal));

        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                             (void*)offsetof(Vertex, texCoords));

        glBindVertexArray(0);
    }

    void draw(GLuint shaderProgram) {
        unsigned int diffuseNr = 1;
        unsigned int specularNr = 1;

        for(unsigned int i = 0; i < textures.size(); i++) {
            glActiveTexture(GL_TEXTURE0 + i);
            std::string number;
            std::string name = textures[i].type;
            if(name == "texture_diffuse")
                number = std::to_string(diffuseNr++);
            else if(name == "texture_specular")
                number = std::to_string(specularNr++);

            glUniform1i(glGetUniformLocation(shaderProgram, (name + number).c_str()), i);
            glBindTexture(GL_TEXTURE_2D, textures[i].id);
        }
        glActiveTexture(GL_TEXTURE0);

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
};
// КРИТЕРИЙ: ТЕНИ
// Глобальные переменные для теней shadow mapping
GLuint depthMapFBO;
GLuint depthMap;
const unsigned int SHADOW_WIDTH = 2048, SHADOW_HEIGHT = 2048;

// Шейдеры
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoords;
out vec4 FragPosLightSpace;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 lightSpaceMatrix;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoords = aTexCoords;
    FragPosLightSpace = lightSpaceMatrix * vec4(FragPos, 1.0);

    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;
in vec4 FragPosLightSpace;

// Текстуры
uniform sampler2D texture_diffuse1;
uniform sampler2D texture_specular1;
uniform sampler2D shadowMap;

// Освещение
struct DirLight {
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct PointLight {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float constant;
    float linear;
    float quadratic;
};

uniform DirLight dirLight;
uniform PointLight pointLights[4];
uniform int numPointLights;

uniform vec3 viewPos;
uniform bool useTextures;
uniform vec3 objectColor;
uniform bool isSmoke;
uniform bool isWindow;
uniform bool isLeaves;
uniform bool hasReflection;
uniform bool isGround;

// Кубическая текстура для отражений
uniform samplerCube skybox;

// Функция расчета теней (улучшенная с PCF)
float ShadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir) {
    // Перспективное деление
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // Преобразование в диапазон [0,1]
    projCoords = projCoords * 0.5 + 0.5;

    if(projCoords.z > 1.0)
        return 0.0;

    // Ближайшая глубина из shadow map
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;

    // Смещение для борьбы с shadow acne
    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);

    // PCF для сглаживания теней
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
        }
    }
    shadow /= 9.0;

    return shadow;
}

// Функции освещения
vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir, float shadow) {
    vec3 lightDir = normalize(-light.direction);

    // Диффузное освещение
    float diff = max(dot(normal, lightDir), 0.0);

    // Отраженное направление
    vec3 reflectDir = reflect(-lightDir, normal);

    // Бликовое освещение
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);

    // Комбинируем результаты
    vec3 ambient = light.ambient;
    vec3 diffuse = light.diffuse * diff;
    vec3 specular = light.specular * spec;

    // Учитываем тень
    vec3 lighting = (ambient + (1.0 - shadow) * (diffuse + specular));
    return lighting;
}

vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
    vec3 lightDir = normalize(light.position - fragPos);

    // Диффузное освещение
    float diff = max(dot(normal, lightDir), 0.0);

    // Отраженное направление
    vec3 reflectDir = reflect(-lightDir, normal);

    // Бликовое освещение
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);

    // Затухание
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));

    // Комбинируем результаты
    vec3 ambient = light.ambient;
    vec3 diffuse = light.diffuse * diff;
    vec3 specular = light.specular * spec;

    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;

    return (ambient + diffuse + specular);
}

void main() {
    if(isSmoke) {
        float alpha = 0.6;
        FragColor = vec4(0.7, 0.7, 0.7, alpha);
        return;
    }

    // Для окон - улучшенная версия с правильной прозрачностью
    if(isWindow) {
        // Используем текстуру стекла если есть, иначе голубой цвет
        vec4 glassColor;
        if(useTextures) {
            glassColor = texture(texture_diffuse1, TexCoords);
            // Смешиваем с голубым для лучшего эффекта стекла
            glassColor = vec4(mix(glassColor.rgb, vec3(0.8, 0.9, 1.0), 0.3), 0.3);
        } else {
            glassColor = vec4(0.8, 0.9, 1.0, 0.3);
        }

        // Слабые отражения от skybox
        if(hasReflection) {
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 reflectDir = reflect(-viewDir, normalize(Normal));
            vec4 reflection = texture(skybox, reflectDir);
            glassColor.rgb = mix(glassColor.rgb, reflection.rgb, 0.2);
        }

        FragColor = glassColor;
        return;
    }

    // Для листьев - альфа-тестирование
    if(isLeaves) {
        if(useTextures) {
            vec4 texColor = texture(texture_diffuse1, TexCoords);
            if(texColor.a < 0.1)
                discard;
            FragColor = texColor;
        } else {
            FragColor = vec4(0.0, 0.5, 0.0, 1.0);
        }
        return;
    }

    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);

    // Расчет теней для направленного света - ТОЛЬКО ДЛЯ ЗЕМЛИ
    float shadow = 0.0;
    if (isGround && textureSize(shadowMap, 0).x > 1) {
        shadow = ShadowCalculation(FragPosLightSpace, norm, normalize(-dirLight.direction));
    }

    vec3 result = CalcDirLight(dirLight, norm, viewDir, shadow);
    for(int i = 0; i < numPointLights; i++)
        result += CalcPointLight(pointLights[i], norm, FragPos, viewDir);

    // Применяем текстуру или цвет
    if(useTextures) {
        vec4 texColor = texture(texture_diffuse1, TexCoords);
        FragColor = vec4(result * texColor.rgb, texColor.a);
    } else {
        FragColor = vec4(result * objectColor, 1.0);
    }
}
)";

// Шейдер для карты глубины (теней)
const char* depthVertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 lightSpaceMatrix;
uniform mat4 model;

void main() {
    gl_Position = lightSpaceMatrix * model * vec4(aPos, 1.0);
}
)";

const char* depthFragmentShaderSource = R"(
#version 330 core
void main() {
}
)";

// Глобальные переменные
GLuint shaderProgram;
GLuint depthShaderProgram;
// КРИТЕРИЙ: БАЗОВАЯ СЦЕНА
std::map<std::string, Mesh> meshes; // Объекты сцены
// КРИТЕРИЙ: ТЕКСТУРЫ
std::map<std::string, GLuint> textures; // Текстуры объектов
// КРИТЕРИЙ: ОТРАЖЕНИЯ
GLuint cubemapTexture; // Cubemap для отражений

// КРИТЕРИЙ: КАМЕРА. УПРАВЛЕНИЕ КАМЕРОЙ
glm::vec3 cameraPos = glm::vec3(0.0f, 3.0f, 10.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
float cameraSpeed = 0.05f;
float yaw = -90.0f;
float pitch = 0.0f;
float lastX = 400, lastY = 300;
bool firstMouse = true;

// КРИТЕРИЙ: АНИМАЦИЯ СЦЕНЫ
float doorAngle = 0.0f; // Анимация двери
bool doorOpening = false;
float animationTime = 0.0f;

// КРИТЕРИЙ: ОСВЕЩЕНИЕ
DirLight dirLight; // источники света
PointLight pointLights[4];
int numPointLights = 4;

// КРИТЕРИЙ: АНИМАЦИЯ СЦЕНЫ
std::vector<SmokeParticle> smokeParticles; // система частиц дыма
const int MAX_SMOKE_PARTICLES = 100;
float smokeSpawnTimer = 0.0f;

// Прототипы функций
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void processInput(GLFWwindow* window);
GLuint loadTexture(const char* path, bool alpha = false);
GLuint loadCubemap(std::vector<std::string> faces);
void createShaderProgram();
void createDepthShaderProgram();
void configureShadowMapping();
Mesh createCubeMesh();
Mesh createPlaneMesh();
Mesh createPyramidMesh();
Mesh createCylinderMesh(int segments = 16);
void setupScene();
void renderScene();
void renderShadowMap();
void updateAnimation(float deltaTime);
void initSmokeParticles();
void updateSmokeParticles(float deltaTime);
void renderSmoke();
void renderTree(glm::vec3 position, float scale = 1.0f);

// КРИТЕРИЙ: ТЕКСТУРЫ
GLuint createDefaultTexture(glm::vec3 color = glm::vec3(0.5f, 0.5f, 0.5f)) {
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    // Создаем простую текстуру 2x2 пикселя указанного цвета
    unsigned char data[16];
    for(int i = 0; i < 16; i += 4) {
        data[i] = (unsigned char)(color.r * 255);
        data[i+1] = (unsigned char)(color.g * 255);
        data[i+2] = (unsigned char)(color.b * 255);
        data[i+3] = 255;
    }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return textureID;
}

GLuint createGrassTexture() { // функция создания травы
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    // Создаем текстуру травы 4x4 пикселя
    unsigned char data[] = {
        34, 139, 34, 255,   46, 139, 87, 255,   34, 139, 34, 255,   46, 139, 87, 255,
        46, 139, 87, 255,   34, 139, 34, 255,   46, 139, 87, 255,   34, 139, 34, 255,
        34, 139, 34, 255,   46, 139, 87, 255,   34, 139, 34, 255,   46, 139, 87, 255,
        46, 139, 87, 255,   34, 139, 34, 255,   46, 139, 87, 255,   34, 139, 34, 255
    };
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    return textureID;
}

GLuint createGlassTexture() { // функция для создания стекла
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    // Создаем текстуру стекла
    unsigned char data[] = {
        200, 220, 240, 100,  210, 230, 250, 120,
        210, 230, 250, 120,  200, 220, 240, 100
    };
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    return textureID;
}
// КРИТЕРИЙ: КАМЕРА. УПРАВЛЕНИЕ КАМЕРОЙ
// Обработка оконных событий
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}
// Функция обработки мыши
void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;

    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);
}
// Обработка ввода с клавиатуры
void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    float cameraSpeed = 2.5f * 0.05f;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        cameraPos -= cameraUp * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        cameraPos += cameraUp * cameraSpeed;
    static bool spacePressed = false; // Управление анимацией двери
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS && !spacePressed) {
        doorOpening = !doorOpening;
        spacePressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_RELEASE) {
        spacePressed = false;
    }
}
// КРИТЕРИЙ: ТЕКСТУРЫ
GLuint loadTexture(const char* path, bool alpha) { // Загрузка текстур из файлов
    GLuint textureID;
    glGenTextures(1, &textureID);
    int width, height, nrComponents;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(path, &width, &height, &nrComponents, 0);
    if (data) {
        GLenum format;
        if (nrComponents == 1)
            format = GL_RED;
        else if (nrComponents == 3)
            format = GL_RGB;
        else if (nrComponents == 4)
            format = GL_RGBA;
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        stbi_image_free(data);
        std::cout << "Текстура загружена: " << path << std::endl;
    } else {
        std::cout << "Текстура не найдена: " << path << ", создаем текстуру по умолчанию" << std::endl;
        if (std::string(path) == "glass.jpg") {
            textureID = createGlassTexture();
        } else if (std::string(path) == "grass.jpg") {
            textureID = createGrassTexture();
        } else {
            textureID = createDefaultTexture();
        }
    }
    return textureID;
}
// КРИТЕРИЙ: ОТРАЖЕНИЯ
GLuint loadCubemap(std::vector<std::string> faces) {
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);
    int width, height, nrChannels;
    for (unsigned int i = 0; i < faces.size(); i++) {
        unsigned char* data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);
        if (data) {
            GLenum format;
            if (nrChannels == 1)
                format = GL_RED;
            else if (nrChannels == 3)
                format = GL_RGB;
            else if (nrChannels == 4)
                format = GL_RGBA;
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, format,
                        width, height, 0, format, GL_UNSIGNED_BYTE, data);
            stbi_image_free(data);
            std::cout << "Skybox texture loaded: " << faces[i] << std::endl;
        } else {
            std::cout << "Cubemap texture failed to load at path: " << faces[i] << std::endl;
            stbi_image_free(data);
            // Создаем простую текстуру по умолчанию если загрузка не удалась
            std::vector<unsigned char> defaultData(512 * 512 * 3, 128); // серый цвет
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB,
                        512, 512, 0, GL_RGB, GL_UNSIGNED_BYTE, defaultData.data());
        }
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    return textureID;
}
// КРИТЕРИЙ: МАТЕРИАЛЫ
void createShaderProgram() { // компиляция шейдеров с освещением
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void createDepthShaderProgram() { // создание шейдеров для глубины теней
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &depthVertexShaderSource, NULL);
    glCompileShader(vertexShader);
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::DEPTH_VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &depthFragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::DEPTH_FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    depthShaderProgram = glCreateProgram();
    glAttachShader(depthShaderProgram, vertexShader);
    glAttachShader(depthShaderProgram, fragmentShader);
    glLinkProgram(depthShaderProgram);
    glGetProgramiv(depthShaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(depthShaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::DEPTH_PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

// КРИТЕРИЙ: ТЕНИ
void configureShadowMapping() { // настройка shadow mapping
    // Создание FBO для shadow map
    glGenFramebuffers(1, &depthMapFBO);
    // Создание текстуры глубины
    glGenTextures(1, &depthMap);
    glBindTexture(GL_TEXTURE_2D, depthMap);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
                SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    // Прикрепление текстуры глубины как буфера глубины FBO
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    // Проверка полноты кадрового буфера
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Shadow framebuffer is not complete!" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
// КРИТЕРИЙ: БАЗОВАЯ СЦЕНА
Mesh createCubeMesh() { // Cоздание куба
    std::vector<Vertex> vertices = {
        // Передняя грань
        {{-0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
        {{ 0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
        {{ 0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
        {{-0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
        // Задняя грань
        {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
        {{-0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
        {{ 0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},
        {{ 0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
        // Левая грань
        {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
        {{-0.5f, -0.5f,  0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
        {{-0.5f,  0.5f,  0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
        {{-0.5f,  0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
        // Правая грань
        {{ 0.5f, -0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
        {{ 0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
        {{ 0.5f,  0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
        {{ 0.5f,  0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
        // Верхняя грань
        {{-0.5f,  0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        {{ 0.5f,  0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
        {{ 0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
        {{-0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
        // Нижняя грань
        {{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
        {{ 0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
        {{ 0.5f, -0.5f,  0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
        {{-0.5f, -0.5f,  0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}}
    };
    std::vector<unsigned int> indices = {
        0, 1, 2, 2, 3, 0,       // перед
        4, 5, 6, 6, 7, 4,       // зад
        8, 9, 10, 10, 11, 8,    // лево
        12, 13, 14, 14, 15, 12, // право
        16, 17, 18, 18, 19, 16, // верх
        20, 21, 22, 22, 23, 20  // низ
    };
    Mesh mesh;
    mesh.vertices = vertices;
    mesh.indices = indices;
    mesh.setupMesh();
    return mesh;
}

Mesh createPlaneMesh() { // Создание плоскости
    std::vector<Vertex> vertices = {
        {{-0.5f, 0.0f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        {{ 0.5f, 0.0f, -0.5f}, {0.0f, 1.0f, 0.0f}, {10.0f, 0.0f}},
        {{ 0.5f, 0.0f,  0.5f}, {0.0f, 1.0f, 0.0f}, {10.0f, 10.0f}},
        {{-0.5f, 0.0f,  0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 10.0f}}
    };
    std::vector<unsigned int> indices = {
        0, 1, 2, 2, 3, 0
    };
    Mesh mesh;
    mesh.vertices = vertices;
    mesh.indices = indices;
    mesh.setupMesh();
    return mesh;
}

Mesh createPyramidMesh() { // Создание пирамиды
    std::vector<Vertex> vertices = {
        // Основание
        {{-0.5f, 0.0f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
        {{ 0.5f, 0.0f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
        {{ 0.5f, 0.0f,  0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
        {{-0.5f, 0.0f,  0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
        // Передняя грань
        {{-0.5f, 0.0f,  0.5f}, {-0.5f, 0.5f,  0.5f}, {0.0f, 0.0f}},
        {{ 0.5f, 0.0f,  0.5f}, { 0.5f, 0.5f,  0.5f}, {1.0f, 0.0f}},
        {{ 0.0f, 1.0f,  0.0f}, { 0.0f, 0.5f,  0.5f}, {0.5f, 1.0f}},
        // Правая грань
        {{ 0.5f, 0.0f,  0.5f}, { 0.5f, 0.5f,  0.5f}, {0.0f, 0.0f}},
        {{ 0.5f, 0.0f, -0.5f}, { 0.5f, 0.5f, -0.5f}, {1.0f, 0.0f}},
        {{ 0.0f, 1.0f,  0.0f}, { 0.0f, 0.5f,  0.0f}, {0.5f, 1.0f}},
        // Задняя грань
        {{ 0.5f, 0.0f, -0.5f}, { 0.5f, 0.5f, -0.5f}, {0.0f, 0.0f}},
        {{-0.5f, 0.0f, -0.5f}, {-0.5f, 0.5f, -0.5f}, {1.0f, 0.0f}},
        {{ 0.0f, 1.0f,  0.0f}, { 0.0f, 0.5f, -0.5f}, {0.5f, 1.0f}},
        // Левая грань
        {{-0.5f, 0.0f, -0.5f}, {-0.5f, 0.5f, -0.5f}, {0.0f, 0.0f}},
        {{-0.5f, 0.0f,  0.5f}, {-0.5f, 0.5f,  0.5f}, {1.0f, 0.0f}},
        {{ 0.0f, 1.0f,  0.0f}, { 0.0f, 0.5f,  0.0f}, {0.5f, 1.0f}}
    };
    std::vector<unsigned int> indices = {
        // Основание
        // 0, 1, 2, 2, 3, 0,
        4, 5, 6,   // передняя
        7, 8, 9,   // правая
        10, 11, 12, // задняя
        13, 14, 15  // левая
    };
    Mesh mesh;
    mesh.vertices = vertices;
    mesh.indices = indices;
    mesh.setupMesh();
    return mesh;
}

Mesh createCylinderMesh(int segments) { // Создание цилиндра
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    // Боковая поверхность
    for (int i = 0; i <= segments; ++i) {
        float angle = 2.0f * M_PI * i / segments;
        float x = cos(angle);
        float z = sin(angle);
        // Вершины для боковой поверхности
        vertices.push_back({{x * 0.5f, -0.5f, z * 0.5f}, {x, 0.0f, z}, {static_cast<float>(i) / segments, 0.0f}});
        vertices.push_back({{x * 0.5f, 0.5f, z * 0.5f}, {x, 0.0f, z}, {static_cast<float>(i) / segments, 1.0f}});
    }
    // Индексы для боковой поверхности
    for (int i = 0; i < segments; ++i) {
        int base = i * 2;
        indices.push_back(base);
        indices.push_back(base + 1);
        indices.push_back(base + 2);
        indices.push_back(base + 1);
        indices.push_back(base + 3);
        indices.push_back(base + 2);
    }
    Mesh mesh;
    mesh.vertices = vertices;
    mesh.indices = indices;
    mesh.setupMesh();
    return mesh;
}
// КРИТЕРИЙ: БАЗОВАЯ СЦЕНА
void setupScene() {
    // Загрузка текстур с обработкой ошибок, настройка объектов сцены
    std::cout << "Загрузка текстур..." << std::endl;
    textures["glass"] = loadTexture("glass.jpg", true);
    textures["wall"] = loadTexture("wall.jpg");
    textures["roof"] = loadTexture("roof.jpg");
    textures["grass"] = loadTexture("grass.jpg");
    textures["brick"] = loadTexture("brick.jpg");
    textures["door"] = loadTexture("door.jpg");
    textures["bark"] = loadTexture("bark.jpg");
    textures["leaves"] = loadTexture("leaves.jpg", true);
    // КРИТЕРИЙ: ОТРАЖЕНИЯ
    // Загрузка skybox текстур из файлов
    std::vector<std::string> faces = {
        "skybox/right.jpg",
        "skybox/left.jpg",
        "skybox/top.jpg",
        "skybox/bottom.jpg",
        "skybox/front.jpg",
        "skybox/back.jpg"
    };
    cubemapTexture = loadCubemap(faces);
    // Создание мешей
    meshes["cube"] = createCubeMesh();
    meshes["plane"] = createPlaneMesh();
    meshes["pyramid"] = createPyramidMesh();
    meshes["cylinder"] = createCylinderMesh(16);
}
// КРИТЕРИЙ: АНИМАЦИЯ СЦЕНЫ
void initSmokeParticles() { // Инициализация системы частиц
    smokeParticles.clear();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1f, 0.1f);
    for (int i = 0; i < MAX_SMOKE_PARTICLES; i++) {
        SmokeParticle particle;
        particle.position = glm::vec3(1.5f, 5.7f, -1.0f);
        particle.velocity = glm::vec3(dis(gen), 0.8f + dis(gen) * 0.2f, dis(gen));
        particle.life = 0.0f;
        particle.scale = 0.1f + static_cast<float>(dis(gen)) * 0.05f;
        smokeParticles.push_back(particle);
    }
}
// КРИТЕРИЙ: АНИМАЦИЯ СЦЕНЫ
void updateAnimation(float deltaTime) {
    // Анимация двери
    if (doorOpening) {
        doorAngle = glm::min(doorAngle + 60.0f * deltaTime, 90.0f);
    } else {
        doorAngle = glm::max(doorAngle - 60.0f * deltaTime, 0.0f);
    }

    animationTime += deltaTime;
}
void updateSmokeParticles(float deltaTime) {
    // Обновление системы частиц дыма
    smokeSpawnTimer += deltaTime;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1f, 0.1f);
    for (auto& particle : smokeParticles) {
        if (particle.life <= 0.0f) {
            if (smokeSpawnTimer >= 0.1f) {
                particle.position = glm::vec3(1.5f, 5.7f, -1.0f);
                particle.velocity = glm::vec3(dis(gen), 0.8f + dis(gen) * 0.2f, dis(gen));
                particle.life = 3.0f;
                particle.scale = 0.1f + static_cast<float>(dis(gen)) * 0.05f;
                smokeSpawnTimer = 0.0f;
            }
        } else {
            particle.position += particle.velocity * deltaTime;
            particle.life -= deltaTime;
            particle.scale += 0.1f * deltaTime;
            particle.velocity.x += dis(gen) * 0.1f;
            particle.velocity.z += dis(gen) * 0.1f;
        }
    }
}
// КРИТЕРИЙ: ПРОЗРАЧНОСТЬ
void renderSmoke() { // Рендер прозрачного дыма
    glUseProgram(shaderProgram);
    glUniform1i(glGetUniformLocation(shaderProgram, "isSmoke"), true);
    glUniform1i(glGetUniformLocation(shaderProgram, "useTextures"), false);
    // Включение смешивания для дыма
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    for (const auto& particle : smokeParticles) {
        if (particle.life > 0.0f) {
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, particle.position);
            model = glm::scale(model, glm::vec3(particle.scale));
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
            meshes["cube"].draw(shaderProgram);
        }
    }
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glUniform1i(glGetUniformLocation(shaderProgram, "isSmoke"), false);
}
// КРИТЕРИЙ: БАЗОВАЯ СЦЕНА
void renderTree(glm::vec3 position, float scale) { // Рендер дерева
    glUseProgram(shaderProgram);
    // Ствол дерева
    glBindTexture(GL_TEXTURE_2D, textures["bark"]);
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, position);
    model = glm::scale(model, glm::vec3(0.3f * scale, 2.0f * scale, 0.3f * scale));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cylinder"].draw(shaderProgram);
    // Листва (сфера из кубов)
    glBindTexture(GL_TEXTURE_2D, textures["leaves"]);
    glUniform1i(glGetUniformLocation(shaderProgram, "isLeaves"), true);
    glUniform1i(glGetUniformLocation(shaderProgram, "useTextures"), true);
    std::vector<glm::vec3> leafPositions = {
        {0.0f, 1.2f * scale, 0.0f},
        {0.4f * scale, 1.5f * scale, 0.2f * scale},
        {-0.3f * scale, 1.6f * scale, -0.2f * scale},
        {0.3f * scale, 1.4f * scale, -0.4f * scale},
        {-0.4f * scale, 1.3f * scale, 0.3f * scale},
        {0.2f * scale, 1.7f * scale, 0.1f * scale},
        {-0.2f * scale, 1.8f * scale, -0.1f * scale}
    };
    for (const auto& leafPos : leafPositions) {
        model = glm::mat4(1.0f);
        model = glm::translate(model, position + leafPos);
        model = glm::scale(model, glm::vec3(0.6f * scale));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        meshes["cube"].draw(shaderProgram);
    }
    glUniform1i(glGetUniformLocation(shaderProgram, "isLeaves"), false);
}
// КРИТЕРИЙ: ТЕНИ
void renderSceneForShadow(glm::mat4 lightSpaceMatrix) { // Рендер сцены для shadow map
    glUseProgram(depthShaderProgram);
    glUniformMatrix4fv(glGetUniformLocation(depthShaderProgram, "lightSpaceMatrix"), 1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));
    // Рендерим основные объекты, которые должны отбрасывать тени
    glm::mat4 model = glm::mat4(1.0f);
    // Основной дом - стены
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, 2.0f, -2.0f));
    model = glm::scale(model, glm::vec3(6.0f, 4.0f, 0.2f));
    glUniformMatrix4fv(glGetUniformLocation(depthShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(depthShaderProgram);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(-3.0f, 2.0f, 0.0f));
    model = glm::scale(model, glm::vec3(0.2f, 4.0f, 4.0f));
    glUniformMatrix4fv(glGetUniformLocation(depthShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(depthShaderProgram);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(3.0f, 2.0f, 0.0f));
    model = glm::scale(model, glm::vec3(0.2f, 4.0f, 4.0f));
    glUniformMatrix4fv(glGetUniformLocation(depthShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(depthShaderProgram);
    // Передняя стена
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(-2.0f, 2.0f, 2.0f));
    model = glm::scale(model, glm::vec3(2.0f, 4.0f, 0.2f));
    glUniformMatrix4fv(glGetUniformLocation(depthShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(depthShaderProgram);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(2.0f, 2.0f, 2.0f));
    model = glm::scale(model, glm::vec3(2.0f, 4.0f, 0.2f));
    glUniformMatrix4fv(glGetUniformLocation(depthShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(depthShaderProgram);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, 3.5f, 2.0f));
    model = glm::scale(model, glm::vec3(2.0f, 1.0f, 0.2f));
    glUniformMatrix4fv(glGetUniformLocation(depthShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(depthShaderProgram);
    // Дверь
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(1.0f, 1.5f, 1.95f));
    model = glm::rotate(model, glm::radians(-doorAngle), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::translate(model, glm::vec3(-1.0f, 0.0f, 0.0f));
    model = glm::scale(model, glm::vec3(2.0f, 3.0f, 0.05f));
    glUniformMatrix4fv(glGetUniformLocation(depthShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(depthShaderProgram);
    // Крыша
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, 4.0f, 0.0f));
    model = glm::scale(model, glm::vec3(6.2f, 2.0f, 4.2f));
    glUniformMatrix4fv(glGetUniformLocation(depthShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["pyramid"].draw(depthShaderProgram);
    // Труба
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(1.5f, 5.2f, -1.0f));
    model = glm::scale(model, glm::vec3(0.4f, 1.0f, 0.4f));
    glUniformMatrix4fv(glGetUniformLocation(depthShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(depthShaderProgram);
    // Деревья (стволы для теней)
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(-8.0f, 1.0f, -6.0f));
    model = glm::scale(model, glm::vec3(0.3f, 2.0f, 0.3f));
    glUniformMatrix4fv(glGetUniformLocation(depthShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cylinder"].draw(depthShaderProgram);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(7.0f, 1.0f, -7.0f));
    model = glm::scale(model, glm::vec3(0.3f, 2.0f, 0.3f));
    glUniformMatrix4fv(glGetUniformLocation(depthShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cylinder"].draw(depthShaderProgram);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(-6.0f, 1.0f, 8.0f));
    model = glm::scale(model, glm::vec3(0.3f, 2.0f, 0.3f));
    glUniformMatrix4fv(glGetUniformLocation(depthShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cylinder"].draw(depthShaderProgram);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(9.0f, 1.0f, 5.0f));
    model = glm::scale(model, glm::vec3(0.3f, 2.0f, 0.3f));
    glUniformMatrix4fv(glGetUniformLocation(depthShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cylinder"].draw(depthShaderProgram);
}
// КРИТЕРИЙ: ТЕНИ
void renderShadowMap() { // Создание shadow map
    glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glClear(GL_DEPTH_BUFFER_BIT);
    // Вычисление матрицы света
    float near_plane = 1.0f, far_plane = 50.0f;
    glm::mat4 lightProjection = glm::ortho(-20.0f, 20.0f, -20.0f, 20.0f, near_plane, far_plane);
    // Свет
    glm::mat4 lightView = glm::lookAt(glm::vec3(-15.0f, 15.0f, 15.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0, 1.0, 0.0));
    glm::mat4 lightSpaceMatrix = lightProjection * lightView;
    // Рендер объектов, которые отбрасывают тени
    renderSceneForShadow(lightSpaceMatrix);
    // Возврат к обычному фреймбуферу и вьюпорту
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, 1200, 800);
}
// КРИТЕРИЙ: БАЗОВАЯ СЦЕНА
void renderScene() { // Основной рендер сцены
    glUseProgram(shaderProgram);
    // Матрицы вида и проекции
    glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1200.0f / 800.0f, 0.1f, 100.0f);
    // КРИТЕРИЙ: ТЕНИ
    float near_plane = 1.0f, far_plane = 25.0f; // Передача параметров теней в шейдер
    glm::mat4 lightProjection = glm::ortho(-15.0f, 15.0f, -15.0f, 15.0f, near_plane, far_plane);
    glm::mat4 lightView = glm::lookAt(glm::vec3(-15.0f, 15.0f, 15.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0, 1.0, 0.0));
    glm::mat4 lightSpaceMatrix = lightProjection * lightView;
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "lightSpaceMatrix"), 1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));
    glUniform3f(glGetUniformLocation(shaderProgram, "viewPos"), cameraPos.x, cameraPos.y, cameraPos.z);
    // Привязка shadow map
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, depthMap);
    glUniform1i(glGetUniformLocation(shaderProgram, "shadowMap"), 2);
    // КРИТЕРИЙ: ОТРАЖЕНИЯ
    // Привязка cubemap для отражений
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
    glUniform1i(glGetUniformLocation(shaderProgram, "skybox"), 3);
    // Включение отражений для окон
    glUniform1i(glGetUniformLocation(shaderProgram, "hasReflection"), true);
    // КРИТЕРИЙ: ОСВЕЩЕНИЕ
    // Настройка параметров освещения
    glm::vec3 lightDir = glm::normalize(glm::vec3(-1.0f, -1.0f, -1.0f)); // Свет с левого-верхнего-переднего угла
    glUniform3fv(glGetUniformLocation(shaderProgram, "dirLight.direction"), 1, glm::value_ptr(lightDir));
    glUniform3fv(glGetUniformLocation(shaderProgram, "dirLight.ambient"), 1, glm::value_ptr(dirLight.ambient));
    glUniform3fv(glGetUniformLocation(shaderProgram, "dirLight.diffuse"), 1, glm::value_ptr(dirLight.diffuse));
    glUniform3fv(glGetUniformLocation(shaderProgram, "dirLight.specular"), 1, glm::value_ptr(dirLight.specular));
    glUniform1i(glGetUniformLocation(shaderProgram, "numPointLights"), numPointLights);
    for(int i = 0; i < numPointLights; i++) {
        std::string prefix = "pointLights[" + std::to_string(i) + "].";
        glUniform3fv(glGetUniformLocation(shaderProgram, (prefix + "position").c_str()), 1, glm::value_ptr(pointLights[i].position));
        glUniform3fv(glGetUniformLocation(shaderProgram, (prefix + "ambient").c_str()), 1, glm::value_ptr(pointLights[i].ambient));
        glUniform3fv(glGetUniformLocation(shaderProgram, (prefix + "diffuse").c_str()), 1, glm::value_ptr(pointLights[i].diffuse));
        glUniform3fv(glGetUniformLocation(shaderProgram, (prefix + "specular").c_str()), 1, glm::value_ptr(pointLights[i].specular));
        glUniform1f(glGetUniformLocation(shaderProgram, (prefix + "constant").c_str()), pointLights[i].constant);
        glUniform1f(glGetUniformLocation(shaderProgram, (prefix + "linear").c_str()), pointLights[i].linear);
        glUniform1f(glGetUniformLocation(shaderProgram, (prefix + "quadratic").c_str()), pointLights[i].quadratic);
    }
    // КРИТЕРИЙ: БАЗОВАЯ СЦЕНА
    // Рендер земли
    glDisable(GL_CULL_FACE);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textures["grass"]);
    glUniform1i(glGetUniformLocation(shaderProgram, "texture_diffuse1"), 0);
    glUniform1i(glGetUniformLocation(shaderProgram, "useTextures"), true);
    glUniform1i(glGetUniformLocation(shaderProgram, "isGround"), true);
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, -0.1f, 0.0f));
    model = glm::scale(model, glm::vec3(100.0f, 1.0f, 100.0f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["plane"].draw(shaderProgram);
    glUniform1i(glGetUniformLocation(shaderProgram, "isGround"), false);
    glEnable(GL_CULL_FACE);
    // Внутренний пол
    glBindTexture(GL_TEXTURE_2D, textures["wall"]);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, 0.01f, 0.0f));
    model = glm::scale(model, glm::vec3(5.8f, 0.1f, 3.8f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(shaderProgram);
    // Потолок
    glBindTexture(GL_TEXTURE_2D, textures["wall"]);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, 3.99f, 0.0f)); // Чуть ниже крыши
    model = glm::scale(model, glm::vec3(5.8f, 0.05f, 3.8f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(shaderProgram);
    // Основное здание - стены
    glBindTexture(GL_TEXTURE_2D, textures["wall"]);
    // Задняя стена с оконным проемом
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, 3.5f, -2.0f));
    model = glm::scale(model, glm::vec3(6.0f, 1.0f, 0.2f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(shaderProgram);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, 0.5f, -2.0f));
    model = glm::scale(model, glm::vec3(6.0f, 1.0f, 0.2f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(shaderProgram);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(-2.5f, 2.0f, -2.0f));
    model = glm::scale(model, glm::vec3(1.0f, 2.0f, 0.2f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(shaderProgram);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(2.5f, 2.0f, -2.0f));
    model = glm::scale(model, glm::vec3(1.0f, 2.0f, 0.2f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(shaderProgram);
    // Левая стена с оконным проемом
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(-3.0f, 3.5f, 0.0f));
    model = glm::scale(model, glm::vec3(0.2f, 1.0f, 4.0f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(shaderProgram);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(-3.0f, 0.5f, 0.0f));
    model = glm::scale(model, glm::vec3(0.2f, 1.0f, 4.0f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(shaderProgram);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(-3.0f, 2.0f, 1.5f));
    model = glm::scale(model, glm::vec3(0.2f, 2.0f, 1.0f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(shaderProgram);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(-3.0f, 2.0f, -1.5f));
    model = glm::scale(model, glm::vec3(0.2f, 2.0f, 1.0f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(shaderProgram);
    // Правая стена с оконным проемом
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(3.0f, 3.5f, 0.0f));
    model = glm::scale(model, glm::vec3(0.2f, 1.0f, 4.0f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(shaderProgram);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(3.0f, 0.5f, 0.0f));
    model = glm::scale(model, glm::vec3(0.2f, 1.0f, 4.0f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(shaderProgram);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(3.0f, 2.0f, 1.5f));
    model = glm::scale(model, glm::vec3(0.2f, 2.0f, 1.0f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(shaderProgram);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(3.0f, 2.0f, -1.5f));
    model = glm::scale(model, glm::vec3(0.2f, 2.0f, 1.0f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(shaderProgram);
    // Передняя стена с дверным проемом
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(-2.0f, 2.0f, 2.0f));
    model = glm::scale(model, glm::vec3(2.0f, 4.0f, 0.2f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(shaderProgram);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(2.0f, 2.0f, 2.0f));
    model = glm::scale(model, glm::vec3(2.0f, 4.0f, 0.2f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(shaderProgram);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, 3.5f, 2.0f));
    model = glm::scale(model, glm::vec3(2.0f, 1.0f, 0.2f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(shaderProgram);
    // КРИТЕРИЙ: АНИМАЦИЯ СЦЕНЫ
    // Анимированная дверь
    glBindTexture(GL_TEXTURE_2D, textures["door"]);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(1.0f, 1.5f, 1.95f));
    model = glm::rotate(model, glm::radians(-doorAngle), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::translate(model, glm::vec3(-1.0f, 0.0f, 0.0f));
    model = glm::scale(model, glm::vec3(2.0f, 3.0f, 0.05f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(shaderProgram);
    // Крыша
    glBindTexture(GL_TEXTURE_2D, textures["roof"]);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, 4.0f, 0.0f));
    model = glm::scale(model, glm::vec3(6.2f, 2.0f, 4.2f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["pyramid"].draw(shaderProgram);
    // Труба
    glBindTexture(GL_TEXTURE_2D, textures["brick"]);
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(1.5f, 5.2f, -1.0f));
    model = glm::scale(model, glm::vec3(0.4f, 1.0f, 0.4f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(shaderProgram);
    // Деревья вокруг дома
    renderTree(glm::vec3(-8.0f, 0.0f, -6.0f), 1.2f);
    renderTree(glm::vec3(7.0f, 0.0f, -7.0f), 1.0f);
    renderTree(glm::vec3(-6.0f, 0.0f, 8.0f), 1.5f);
    renderTree(glm::vec3(9.0f, 0.0f, 5.0f), 1.3f);
    renderTree(glm::vec3(-10.0f, 0.0f, 3.0f), 1.1f);
    renderTree(glm::vec3(5.0f, 0.0f, -10.0f), 1.4f);
    // КРИТЕРИЙ: ПРОЗРАЧНОСТЬ
    // Прозрачные окна
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBindTexture(GL_TEXTURE_2D, textures["glass"]);
    glUniform1i(glGetUniformLocation(shaderProgram, "isWindow"), true);
    glUniform1i(glGetUniformLocation(shaderProgram, "useTextures"), true);
    // Окно на задней стене
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, 2.0f, -2.01f));
    model = glm::scale(model, glm::vec3(4.0f, 2.0f, 0.05f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(shaderProgram);
    // Окно на левой стене
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(-3.01f, 2.0f, 0.0f));
    model = glm::scale(model, glm::vec3(0.05f, 2.0f, 2.0f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(shaderProgram);
    // Окно на правой стене
    model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(3.01f, 2.0f, 0.0f));
    model = glm::scale(model, glm::vec3(0.05f, 2.0f, 2.0f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    meshes["cube"].draw(shaderProgram);
    glUniform1i(glGetUniformLocation(shaderProgram, "isWindow"), false);
    glDisable(GL_BLEND);
}

// ГЛАВНАЯ ФУНКЦИЯ
int main() {
    // Инициализация GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // Создание окна
    GLFWwindow* window = glfwCreateWindow(1200, 800, "Деревенский Дом", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    // Инициализация GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }
    // Настройка OpenGL
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
    // Компиляция шейдеров
    createShaderProgram();
    createDepthShaderProgram();
    // Настройка shadow mapping
    configureShadowMapping();
    // Настройка сцены
    setupScene();
    // Инициализация частиц дыма
    initSmokeParticles();
    // КРИТЕРИЙ: ОСВЕЩЕНИЕ
    // Настройка освещения
    dirLight.direction = glm::vec3(-1.0f, -1.0f, -1.0f);
    dirLight.ambient = glm::vec3(0.4f, 0.4f, 0.4f);
    dirLight.diffuse = glm::vec3(0.8f, 0.8f, 0.8f);
    dirLight.specular = glm::vec3(0.5f, 0.5f, 0.5f);
    // Внутренний свет дома
    pointLights[0].position = glm::vec3(0.0f, 2.5f, 0.0f);
    pointLights[0].ambient = glm::vec3(0.4f, 0.4f, 0.3f);
    pointLights[0].diffuse = glm::vec3(1.0f, 1.0f, 0.8f);
    pointLights[0].specular = glm::vec3(1.0f, 1.0f, 0.9f);
    pointLights[0].constant = 1.0f;
    pointLights[0].linear = 0.09f;
    pointLights[0].quadratic = 0.032f;
    // Наружное освещение
    pointLights[1].position = glm::vec3(-2.0f, 3.0f, 3.0f);
    pointLights[1].ambient = glm::vec3(0.2f, 0.2f, 0.2f);
    pointLights[1].diffuse = glm::vec3(0.8f, 0.8f, 0.6f);
    pointLights[1].specular = glm::vec3(0.8f, 0.8f, 0.6f);
    pointLights[1].constant = 1.0f;
    pointLights[1].linear = 0.07f;
    pointLights[1].quadratic = 0.017f;
    // Свет из окон
    pointLights[2].position = glm::vec3(-3.0f, 2.0f, 0.0f);
    pointLights[2].ambient = glm::vec3(0.3f, 0.3f, 0.2f);
    pointLights[2].diffuse = glm::vec3(0.9f, 0.9f, 0.7f);
    pointLights[2].specular = glm::vec3(0.8f, 0.8f, 0.6f);
    pointLights[2].constant = 1.0f;
    pointLights[2].linear = 0.09f;
    pointLights[2].quadratic = 0.032f;
    pointLights[3].position = glm::vec3(0.0f, 2.0f, -2.0f);
    pointLights[3].ambient = glm::vec3(0.3f, 0.3f, 0.2f);
    pointLights[3].diffuse = glm::vec3(0.9f, 0.9f, 0.7f);
    pointLights[3].specular = glm::vec3(0.8f, 0.8f, 0.6f);
    pointLights[3].constant = 1.0f;
    pointLights[3].linear = 0.09f;
    pointLights[3].quadratic = 0.032f;
    std::cout << "=== Деревенский Дом - Улучшенная Версия с Тенями и Отражениями ===" << std::endl;
    std::cout << "Управление:" << std::endl;
    std::cout << "WASD - движение камеры" << std::endl;
    std::cout << "QE - подъем/опускание камеры" << std::endl;
    std::cout << "Мышь - вращение камеры" << std::endl;
    std::cout << "Пробел - открыть/закрыть дверь" << std::endl;
    std::cout << "ESC - выход" << std::endl;
    // Главный цикл
    float lastFrame = 0.0f;
    while (!glfwWindowShouldClose(window)) {
        float currentFrame = glfwGetTime();
        float deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        processInput(window);
        // КРИТЕРИЙ: АНИМАЦИЯ СЦЕНЫ
        updateAnimation(deltaTime); // Обновление анимации
        updateSmokeParticles(deltaTime);
        // Рендер shadow map
        renderShadowMap();
        // Основной рендер
        glClearColor(0.53f, 0.81f, 0.92f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        renderScene();
        renderSmoke();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    // Освобождение ресурсов
    for(auto& mesh : meshes) {
        glDeleteVertexArrays(1, &mesh.second.VAO);
        glDeleteBuffers(1, &mesh.second.VBO);
        glDeleteBuffers(1, &mesh.second.EBO);
    }
    for(auto& texture : textures) {
        glDeleteTextures(1, &texture.second);
    }
    glDeleteTextures(1, &depthMap);
    glDeleteFramebuffers(1, &depthMapFBO);
    glDeleteProgram(shaderProgram);
    glDeleteProgram(depthShaderProgram);
    glfwTerminate();
    return 0;
}
