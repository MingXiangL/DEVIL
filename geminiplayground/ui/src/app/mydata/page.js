"use client";
import {Button} from "@/components/ui/button"
import {FileIcon, GitHubLogoIcon, PlusIcon, TrashIcon} from "@radix-ui/react-icons"
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuLabel,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {forwardRef, useImperativeHandle, useRef} from "react";
import FileUploadForm from "@/app/mydata/FileUploadForm";
import CodeRepoForm from "@/app/mydata/CodeRepoForm";
import {useMutation, useQuery, useQueryClient} from "@tanstack/react-query";
import {MoreHorizontal} from "lucide-react";
import {axiosInstance} from "@/app/axios";
import DataTable from "@/components/DataTable";
import ConfirmDialog from "@/components/ConfirmDialog";
import {Badge} from "@/components/ui/badge"
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip"


const FilesTable = forwardRef(function FilesTable({...rest}, ref) {

    const confirmDialogRef = useRef();
    const queryClient = useQueryClient()

    const {data, isLoading, refetch} = useQuery({
        queryKey: ["parts"],
        refetchInterval: 10000,
        queryFn: async () => {
            const response = await axiosInstance.get("/parts")
            return response.data
        }
    })

    const deleteFileMutation = useMutation({
        mutationFn: async (name) => {
            await axiosInstance.delete(`/parts/${name}`);
        }
    })
    const refresh = async () => {
        await queryClient.invalidateQueries({queryKey: ["parts"]});
        await refetch();
    }

    useImperativeHandle(ref, () => ({
        refresh
    }), []);


    const filesTableColumns = [{
        header: "Name",
        accessorKey: "name",
    },
        {
            header: "Type",
            accessorKey: "type",
        },
        {
            header: "Status",
            cell: ({row}) => {
                const status = row.original.status
                const statusMessage = row.original.statusMessage
                if (status === "ready" || status === "pending") {
                    return <Badge variant="secondary">{status}</Badge>
                }
                return (
                    <TooltipProvider>
                        <Tooltip>
                            <TooltipTrigger>
                                <Badge variant="destructive"
                                       style={{cursor: "pointer"}}>{status}</Badge>
                            </TooltipTrigger>
                            <TooltipContent>
                                <p>{statusMessage}</p>
                            </TooltipContent>
                        </Tooltip>
                    </TooltipProvider>
                )
            }
        },
        {
            header: "Actions",
            cell: ({row}) => {
                const data = row.original
                return (
                    <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                            <Button variant="ghost" className="h-8 w-8 p-0">
                                <span className="sr-only">Open menu</span>
                                <MoreHorizontal className="h-4 w-4"/>
                            </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                            <DropdownMenuLabel>Actions</DropdownMenuLabel>
                            <DropdownMenuItem
                                onClick={() => {
                                    const dialog = confirmDialogRef.current
                                    if (dialog) {
                                        dialog.open(
                                            "Delete",
                                            "Are you sure you want to delete this file?",
                                            async (result) => {
                                                if (result) {
                                                    await deleteFileMutation.mutateAsync(data.name);
                                                    await queryClient.refetchQueries(["parts"])
                                                }
                                            }
                                        )
                                    }
                                }}
                            >
                                <TrashIcon className="mr-2 h-4 w-4"/> Delete
                            </DropdownMenuItem>
                            {/*<DropdownMenuSeparator/>*/}
                        </DropdownMenuContent>
                    </DropdownMenu>
                )
            }
        }
    ];
    const filesTableData = data ? data.map((item) => {
        return {
            name: item.name,
            type: item.content_type,
            status: item.status,
            statusMessage: item.status_message
        }
    }) : []
    return (
        <>
            <ConfirmDialog ref={confirmDialogRef}/>
            <DataTable columns={filesTableColumns} data={filesTableData} className="w-full"/>
        </>
    );
});


export default function FilesPage() {

    const uploadFileFormRef = useRef();
    const codeRepoFormRef = useRef();
    const fileTableRef = useRef();


    const newFileHandler = () => {
        const form = uploadFileFormRef.current;
        if (form) {
            form.reset();
            form.open();
        }
    }
    const newCodeRepoHandler = () => {
        const form = codeRepoFormRef.current;
        if (form) {
            form.reset();
            form.open();
        }
    }

    return (
        <>
            <header className="w-full p-2">
                <h1 className="text-xl font-semibold">Files</h1>
            </header>
            <main>
                <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                        <Button>
                            <PlusIcon className="mr-2 h-4 w-4"/> Upload Data
                        </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent className="w-56">
                        <DropdownMenuItem onSelect={newFileHandler}> <FileIcon
                            className="mr-2 h-4 w-4"/>Upload File</DropdownMenuItem>
                        <DropdownMenuItem onSelect={newCodeRepoHandler}> <GitHubLogoIcon className="mr-2 h-4 w-4"/> Code
                            repository</DropdownMenuItem>
                    </DropdownMenuContent>
                </DropdownMenu>
                <div className="rounded-md border m-3">
                    <FilesTable ref={fileTableRef}/>
                </div>
            </main>

            <FileUploadForm ref={uploadFileFormRef}/>
            <CodeRepoForm ref={codeRepoFormRef}/>
        </>
    );
}
