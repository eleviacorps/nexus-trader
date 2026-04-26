FROM node:20-alpine AS build
WORKDIR /app/ui/frontend
COPY ui/frontend/package*.json ./
RUN npm ci
COPY ui/frontend/ ./
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/ui/frontend/dist /usr/share/nginx/html
EXPOSE 80
