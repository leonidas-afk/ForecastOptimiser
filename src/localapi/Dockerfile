# use node 20 image
FROM node:20

EXPOSE 3000

WORKDIR /app

COPY . .

RUN npm install

ENTRYPOINT ["npm", "run", "dev"]

