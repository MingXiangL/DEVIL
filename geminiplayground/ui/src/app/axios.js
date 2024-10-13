import axios from "axios";

const API_BASE_URL = `http://${process.env.NEXT_PUBLIC_API_BASE_URL}`;
console.log(API_BASE_URL);

export const axiosInstance = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        "Content-type": "application/json",
    }
});