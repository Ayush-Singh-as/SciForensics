/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The API base is read at build time so the same image can point at a local
  // backend or a deployed one without a code change.
  env: {
    SCIFORENSICS_API: process.env.SCIFORENSICS_API ?? "http://127.0.0.1:8000",
  },
};
export default nextConfig;
