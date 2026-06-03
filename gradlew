#!/bin/sh
# Gradle wrapper stub for Sync Vision
# On your development machine, run: gradle wrapper --gradle-version 8.11.1
# This will generate the proper gradlew and gradle-wrapper.jar

APP_BASE_NAME=${0##*/}
APP_HOME=$( cd "${0%/*}" && pwd )
CLASSPATH=$APP_HOME/gradle/wrapper/gradle-wrapper.jar

# Check for JAVA_HOME
if [ -z "$JAVA_HOME" ]; then
    JAVACMD=java
else
    JAVACMD="$JAVA_HOME/bin/java"
fi

exec "$JAVACMD" \
    -classpath "$CLASSPATH" \
    org.gradle.wrapper.GradleWrapperMain \
    "$@"
