# Use official MySQL image
FROM mysql:8.0

# Set environment variables (can override at runtime)
ENV MYSQL_ROOT_PASSWORD=
ENV MYSQL_DATABASE=
ENV MYSQL_USER=
ENV MYSQL_PASSWORD=

# Copy initialization script into docker-entrypoint-initdb.d
COPY init.sql /docker-entrypoint-initdb.d/
