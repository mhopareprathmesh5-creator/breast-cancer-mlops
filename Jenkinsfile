// CI/CD pipeline for the breast cancer survival model.
// Jenkins (Windows service) reads this file from GitHub and runs it.
// Each stage must succeed before the next one starts; any failure stops the
// pipeline, so a broken change never reaches the deployed API.

pipeline {
    agent any   // run on the Jenkins machine itself (our Windows PC)

    // Trigger: check GitHub for new commits every ~5 minutes ("H" spreads
    // the exact minute). A webhook would be instant, but GitHub can't reach
    // localhost.
    triggers {
        pollSCM('H/5 * * * *')
    }

    options {
        skipDefaultCheckout(true)      // we check out in our own "Checkout" stage
        disableConcurrentBuilds()      // never two deployments at the same time
        timeout(time: 30, unit: 'MINUTES')
    }

    // Environment variables available to every step (%NAME% in bat).
    environment {
        IMAGE = 'breast-cancer-mlops'
        COMPOSE_PROJECT_NAME = 'breast-cancer-mlops'   // same stack name every run
        NETWORK = 'breast-cancer-mlops_default'        // compose's network for this project
        DATA_SOURCE = 'D:\\breast-cancer-mlops\\data\\raw\\Breast_Cancer.csv'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm   // the commit that triggered this build
            }
        }

        stage('Prepare data') {
            // The dataset isn't in Git, so copy it into the workspace.
            steps {
                bat 'if not exist data\\raw mkdir data\\raw'
                bat 'copy /Y "%DATA_SOURCE%" data\\raw\\'
            }
        }

        stage('Build image') {
            // Tag with the build number (traceable) and "latest" (used by compose).
            steps {
                bat 'docker build -t %IMAGE%:%BUILD_NUMBER% -t %IMAGE%:latest .'
            }
        }

        stage('Test') {
            // Run pytest INSIDE the image we just built: we test exactly what
            // we deploy. The workspace (code + tests + data) is mounted at /work.
            steps {
                bat 'docker run --rm -v "%WORKSPACE%:/work" -w /work %IMAGE%:%BUILD_NUMBER% python -m pytest --junitxml=reports/junit.xml'
            }
            post {
                always {
                    junit 'reports/junit.xml'   // test results shown in Jenkins
                }
            }
        }

        stage('Start MLflow') {
            // --wait: return only when the healthcheck says "healthy".
            steps {
                bat 'docker compose up -d --wait mlflow'
            }
        }

        stage('Train') {
            // One-off training container: registers a new version as @champion.
            // A non-zero exit (failed training) fails this stage.
            steps {
                bat 'docker compose run --rm train'
            }
        }

        stage('Deploy API') {
            // Recreate the API so it loads the new champion; --no-deps: don't
            // rerun training; --wait: until /health is healthy.
            steps {
                bat 'docker compose up -d --wait --no-deps --force-recreate api'
            }
        }

        stage('Smoke test') {
            // Call the deployed API over the compose network.
            steps {
                bat 'docker run --rm --network %NETWORK% -v "%WORKSPACE%:/work" -w /work %IMAGE%:%BUILD_NUMBER% python scripts/smoke_check.py http://api:8000'
            }
        }
    }

    post {
        success {
            echo "Deployed build ${env.BUILD_NUMBER}: API at http://localhost:8000"
        }
        failure {
            // Show recent container logs to help diagnose the failure.
            bat 'docker compose logs --tail 50'
        }
    }
}
