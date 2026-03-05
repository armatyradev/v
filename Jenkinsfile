pipeline {
    agent any

    environment {
        HF_HOME = "${WORKSPACE}/.cache/huggingface"
        PIP_CACHE_DIR = "${WORKSPACE}/.cache/pip"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install req') {
            steps {
                sh '''
                python3 -m venv venv
                source venv/bin/activate
                pip install --upgrade pip
                pip install -r requirements_list.txt
                '''
            }
        }

        stage('Run Tests') {
            steps {
                sh '''
                source venv/bin/activate
                pytest new_test.py --junitxml=results.xml
                '''
            }
        }
    }
    post {
        always {
            junit 'results.xml'
            archiveArtifacts artifacts: 'output/*.png', allowEmptyArchive: true
        }
        success {
            echo "Модель успешно работает и функционирует"
        }
        failure {
            echo 'Что-то пошло не так. Чекни логи'
        }
    }
}

