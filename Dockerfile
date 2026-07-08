# syntax=docker/dockerfile:1.7

FROM node:20-alpine AS frontend-builder
WORKDIR /app/investment-ui
COPY investment-ui/package*.json ./
RUN npm ci
COPY investment-ui/ ./
RUN npm run build

FROM python:3.11-slim AS backend-runtime
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir gunicorn
COPY backend/ ./backend
EXPOSE 8848
CMD ["gunicorn", "--bind", "0.0.0.0:8848", "--workers", "2", "--threads", "4", "--timeout", "120", "backend.main:app"]

FROM nginx:1.27-alpine AS frontend-runtime
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=frontend-builder /app/investment-ui/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
