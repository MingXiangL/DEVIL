"use client";
import "./globals.css";
import { Inter as FontSans } from "next/font/google"
import { ThemeProvider as NextThemesProvider } from "next-themes"
import { cn } from "@/lib/utils"
import {TooltipProvider} from "@/components/ui/tooltip";
import DefaultLayout from "@/components/Layouts/DefaultLayout";
import {QueryClient, QueryClientProvider} from "@tanstack/react-query";
import Head from 'next/head';

const fontSans = FontSans({
  subsets: ["latin"],
  variable: "--font-sans",
})

const queryClient = new QueryClient();

// queryClient.getQueryCache().subscribe((event) => {
//   if (event?.type === "observerResultsUpdated") {
//     console.log("invalidate query");
//     event.query?.invalidate();
//   }
// });


function ThemeProvider({ children, ...props }) {
    return <NextThemesProvider {...props}>{children}</NextThemesProvider>
}
const Provider = ({ children }) => {
  return (
      <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          disableTransitionOnChange
      >
      <QueryClientProvider client={queryClient}>
         <TooltipProvider>
           {children}
         </TooltipProvider>
      </QueryClientProvider>
        </ThemeProvider>
  );
};



export default function RootLayout({ children }) {
  return (
    <html lang="en">
    <Head>
        <meta charSet="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <link rel="shortcut icon" href="/favicon.ico"/>
    </Head>
    <body className={cn(
        "min-h-screen bg-background font-sans antialiased",
          fontSans.variable
      )}>
        <Provider>
          <DefaultLayout>
          {children}
          </DefaultLayout>
        </Provider>
      </body>
    </html>
  );
}
