# Frontend image: multi-stage so node_modules and the toolchain do not ship.
FROM node:22-alpine AS deps
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci

FROM node:22-alpine AS build
WORKDIR /web
COPY --from=deps /web/node_modules ./node_modules
COPY web/ ./
# Baked in at build time: Next inlines `env` values, so this cannot be changed
# by a runtime variable. Override with --build-arg for a deployed backend.
ARG SCIFORENSICS_API=http://127.0.0.1:8000
ENV SCIFORENSICS_API=$SCIFORENSICS_API
RUN npm run build

FROM node:22-alpine AS run
WORKDIR /web
ENV NODE_ENV=production
COPY --from=build /web/.next ./.next
COPY --from=build /web/node_modules ./node_modules
COPY --from=build /web/package.json ./
USER node
EXPOSE 3000
CMD ["npm", "start"]
