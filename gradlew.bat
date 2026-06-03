@rem Gradle wrapper stub for Sync Vision
@rem On your development machine, run: gradle wrapper --gradle-version 8.11.1

@if "%DEBUG%"=="" @echo off
set DEFAULT_JAVA_OPTS="-Xmx4096m"
set JAVA_OPTS=%JAVA_OPTS% %DEFAULT_JAVA_OPTS%
set CLASSPATH=%~dp0gradle\wrapper\gradle-wrapper.jar

java %JAVA_OPTS% -classpath "%CLASSPATH%" org.gradle.wrapper.GradleWrapperMain %*
